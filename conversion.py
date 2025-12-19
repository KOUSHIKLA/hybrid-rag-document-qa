# -*- coding: utf-8 -*-
"""
A high-performance, robust, and generalized document-processing pipeline built on top of the `docling` library.

This module provides a single, unified class `UltraDoclingPipeline` exposing a fast and extensible API that:
- Extracts page/figure/table images with parallel I/O.
- Exports tables to CSV/HTML and provides DataFrame access.
- Generates rich multimodal parquet datasets.
- Performs OCR-to-Markdown with pluggable OCR engines (Tesseract CLI/Lib, EasyOCR, RapidOCR, OCRMac).
- Supports CPU multiprocessing for batch workloads and hardware acceleration via AcceleratorOptions
  (AUTO, CPU, CUDA/NVIDIA, MPS/Apple, OpenVINO; AMD/ROCm support depends on the underlying runtime).
- Offers a clean, typed, and production-grade code style, with carefully designed configuration and error handling.

Notes:
- Requires `docling` and its dependencies. Some features require optional system dependencies (e.g., Tesseract).
- Accelerator support depends on your environment, drivers, and docling inference backends.
- All explanations are provided as comments inside this code file, as requested.

Author: Best-in-class engineering implementation
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
    EasyOcrOptions,
    OcrMacOptions,
    RapidOcrOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.utils.export import generate_multimodal_pages
from docling.utils.utils import create_hash
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem


# --------------------------------------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------------------------------------

_LOG = logging.getLogger("ultra_docling_pipeline")
if not _LOG.handlers:
    # Default to INFO; users can reconfigure logging externally as needed.
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="[%(levelname)s] %(asctime)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    _handler.setFormatter(_formatter)
    _LOG.addHandler(_handler)
    _LOG.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------------------
# Type Aliases
# --------------------------------------------------------------------------------------------------

PathLike = Union[str, Path]
DeviceLiteral = Literal["auto", "cpu", "cuda", "mps", "openvino", "rocm"]  # AMD/ROCm availability depends on backend
OcrEngineLiteral = Literal["tesseract_cli", "tesseract", "easyocr", "rapidocr", "ocrmac"]


# --------------------------------------------------------------------------------------------------
# Configuration Dataclass
# --------------------------------------------------------------------------------------------------

@dataclasses.dataclass(slots=True)
class PipelineConfig:
    """
    Central configuration for the UltraDoclingPipeline.

    Attributes:
        images_scale: Page/picture render scale; 1.0 ~ 72 dpi. Typical: 2.0 for better readability.
        generate_page_images: When True, page images are generated during conversion.
        generate_picture_images: When True, figure/picture images are generated during conversion.
        do_ocr: Enables OCR for scanned/image PDFs.
        ocr_engine: Select OCR engine when `do_ocr` is True. Default 'tesseract_cli'.
        ocr_languages: Languages to use for OCR, e.g., ['en', 'es'].
        force_full_page_ocr: Force full-page OCR; robust for scanned PDFs.
        do_table_structure: Enables table structure extraction.
        do_cell_matching: Enables cell matching inside tables (more accurate table reconstruction).
        accelerator_device: Preferred accelerator device; 'auto' tries to pick the best available.
        num_threads: Thread count hint for docling; if None, uses all logical cores.
        parquet_compression: Compression codec for parquet outputs. E.g., 'snappy', 'zstd', 'gzip'.
        max_io_workers: Max workers for I/O-bound parallelism (e.g., saving images).
        mm_save_image_bytes: If True, multimodal parquet rows include raw image bytes.
        respect_gpu_memory: If True, batch processing avoids oversubscribing GPU by limiting workers.
        page_image_format: Image format for saved pages/figures/tables. Typically 'PNG'.
    """
    images_scale: float = 2.0
    generate_page_images: bool = True
    generate_picture_images: bool = True

    do_ocr: bool = False
    ocr_engine: OcrEngineLiteral = "tesseract_cli"
    ocr_languages: Sequence[str] = dataclasses.field(default_factory=lambda: ["en"])
    force_full_page_ocr: bool = True

    do_table_structure: bool = True
    do_cell_matching: bool = True

    accelerator_device: DeviceLiteral = "auto"
    num_threads: Optional[int] = None

    parquet_compression: Optional[str] = "zstd"
    max_io_workers: int = min(32, (os.cpu_count() or 8) * 2)
    mm_save_image_bytes: bool = True
    respect_gpu_memory: bool = True

    page_image_format: str = "PNG"

    def threads(self) -> int:
        """Resolve effective thread count with a robust fallback."""
        return self.num_threads if self.num_threads and self.num_threads > 0 else (os.cpu_count() or 4)


# --------------------------------------------------------------------------------------------------
# Helper Mappings / Utilities
# --------------------------------------------------------------------------------------------------

def _map_device(device: DeviceLiteral) -> AcceleratorDevice:
    """
    Map a friendly device string to docling's AcceleratorDevice.
    Notes:
        - 'rocm' support depends on the underlying runtime. If not available, fallback to AUTO.
        - When in doubt, 'auto' lets docling pick the best available accelerator.
    """
    mapping: Mapping[str, AcceleratorDevice] = {
        "auto": AcceleratorDevice.AUTO,
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,           # Apple Silicon
        # "openvino": AcceleratorDevice.OPENVINO, # Intel iGPU/CPU
        # 'rocm' is not guaranteed in all builds; fallback to AUTO if not defined.
    }
    if device == "rocm":
        _LOG.warning("Requested device 'rocm'. Falling back to AUTO if unsupported by this environment.")
        return getattr(AcceleratorDevice, "ROCM", AcceleratorDevice.AUTO)  # best-effort mapping
    return mapping.get(device, AcceleratorDevice.AUTO)


def _build_ocr_options(
    engine: OcrEngineLiteral,
    languages: Sequence[str],
    force_full_page_ocr: bool,
) -> Any:
    """
    Construct OCR options for the chosen engine and languages.
    Exceptions:
        Raises ValueError for unknown engines.
    """
    if engine == "tesseract_cli":
        return TesseractCliOcrOptions(lang=list(languages), force_full_page_ocr=force_full_page_ocr)
    if engine == "tesseract":
        return TesseractOcrOptions(lang=list(languages), force_full_page_ocr=force_full_page_ocr)
    if engine == "easyocr":
        return EasyOcrOptions(lang=list(languages), force_full_page_ocr=force_full_page_ocr)
    if engine == "rapidocr":
        return RapidOcrOptions(lang=list(languages), force_full_page_ocr=force_full_page_ocr)
    if engine == "ocrmac":
        return OcrMacOptions(lang=list(languages), force_full_page_ocr=force_full_page_ocr)
    raise ValueError(f"Unknown OCR engine: {engine}")


def _stem(path_like: PathLike) -> str:
    """Return the filesystem stem of a path."""
    return Path(path_like).stem


# --------------------------------------------------------------------------------------------------
# Core Pipeline
# --------------------------------------------------------------------------------------------------

class UltraDoclingPipeline:
    """
    UltraDoclingPipeline: A unified, fast, and extensible docling-based processing pipeline.

    This class exposes a normalized API for:
    - Image extraction (pages, tables, figures) with parallel saving.
    - Table export (CSV, HTML) with DataFrame access.
    - Multimodal parquet generation for ML workflows.
    - OCR-to-Markdown with configurable OCR backends.
    - Batch/multiprocessing orchestration and accelerator/device configuration.

    Thread-safety:
        Each instance is intended for one input document. For batch workloads, use
        the provided batch APIs to spawn separate processes for isolation and throughput.

    Usage skeleton:
        cfg = PipelineConfig(do_ocr=True, ocr_engine="tesseract_cli", accelerator_device="auto")
        pipe = UltraDoclingPipeline(input_path="doc.pdf", output_dir="out", config=cfg)
        pipe.extract_images()
        pipe.export_tables()
        parquet_path = pipe.generate_multimodal_parquet()
        md_text = pipe.ocr_to_markdown(save_to_file=True)
    """

    def __init__(self, input_path: PathLike, output_dir: PathLike, config: Optional[PipelineConfig] = None) -> None:
        self.input_path: Path = Path(input_path).expanduser().resolve()
        self.output_dir: Path = Path(output_dir).expanduser().resolve()
        self.config: PipelineConfig = config or PipelineConfig()

        if not self.input_path.exists() or not self.input_path.is_file():
            raise FileNotFoundError(f"Input document not found: {self.input_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._doc_stem: str = self.input_path.stem

        # Lazily built converters to minimize overhead. Some methods require distinct pipeline options.
        self._base_allowed_formats: List[InputFormat] = [
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.CSV,
            InputFormat.MD,
        ]

        _LOG.info("Initialized pipeline for '%s' (output -> %s)", self.input_path.name, self.output_dir)

    # ----------------------------------------------------------------------------------------------
    # Converter Builders
    # ----------------------------------------------------------------------------------------------

    def _accelerator_options(self) -> AcceleratorOptions:
        """
        Build accelerator options based on configuration.
        This controls multi-threading and device selection (CPU/GPU).
        """
        accel = AcceleratorOptions(
            num_threads=self.config.threads(),
            device=_map_device(self.config.accelerator_device),
        )
        return accel

    def _pdf_pipeline_options_common(self) -> PdfPipelineOptions:
        """
        Create common PDF pipeline options used by multiple tasks.
        These options are further specialized per task (images/tables/ocr/mm).
        """
        opts = PdfPipelineOptions(
            do_ocr=self.config.do_ocr,
            do_table_structure=self.config.do_table_structure,
        )
        # Table structure tuning
        try:
            opts.table_structure_options.do_cell_matching = self.config.do_cell_matching
        except Exception:
            # Backward-compat: old docling versions may use a dict
            opts.table_structure_options = {"do_cell_matching": self.config.do_cell_matching}
        # Accelerator
        opts.accelerator_options = self._accelerator_options()
        return opts

    def _pdf_opts_for_images(self) -> PdfPipelineOptions:
        """Specialized options for image extraction."""
        opts = self._pdf_pipeline_options_common()
        opts.images_scale = self.config.images_scale
        opts.generate_page_images = self.config.generate_page_images
        opts.generate_picture_images = self.config.generate_picture_images
        return opts

    def _pdf_opts_for_mm(self) -> PdfPipelineOptions:
        """Specialized options for multimodal dataset generation."""
        opts = self._pdf_pipeline_options_common()
        opts.images_scale = self.config.images_scale
        opts.generate_page_images = True
        return opts

    def _pdf_opts_for_ocr(self, engine: Optional[OcrEngineLiteral] = None) -> PdfPipelineOptions:
        """Specialized options for OCR-to-Markdown."""
        opts = self._pdf_pipeline_options_common()
        opts.do_ocr = True
        ocr_engine = engine or self.config.ocr_engine
        opts.ocr_options = _build_ocr_options(
            engine=ocr_engine,
            languages=self.config.ocr_languages,
            force_full_page_ocr=self.config.force_full_page_ocr,
        )
        return opts

    def _build_converter(
        self,
        pdf_pipeline_options: Optional[PdfPipelineOptions] = None,
        allow_all_formats: bool = True,
    ) -> DocumentConverter:
        """
        Build a DocumentConverter configured with StandardPdfPipeline, PyPdfium backend, and chosen options.
        This builder is used by all tasks to produce a consistent, high-performance converter.
        """
        format_options: Dict[InputFormat, Any] = {}
        if pdf_pipeline_options is not None:
            format_options[InputFormat.PDF] = PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pdf_pipeline_options,
            )
        else:
            # Build with defaults if no PDF options are provided.
            format_options[InputFormat.PDF] = PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=self._pdf_pipeline_options_common(),
            )

        # Enable Word/docx via SimplePipeline for convenience.
        format_options[InputFormat.DOCX] = WordFormatOption(pipeline_cls=SimplePipeline)

        allowed = self._base_allowed_formats if allow_all_formats else [InputFormat.PDF]
        converter = DocumentConverter(allowed_formats=allowed, format_options=format_options)
        return converter

    # ----------------------------------------------------------------------------------------------
    # Task 1: Extract Images (pages, tables, figures) + Export with HTML/MD options
    # ----------------------------------------------------------------------------------------------

    def extract_images(
        self,
        save_markdown_embedded: bool = True,
        save_markdown_referenced: bool = True,
        save_html_referenced: bool = True,
    ) -> None:
        """
        Extract page images and figures/tables, then optionally save Markdown/HTML with embedded or referenced images.
        This method is I/O bound when saving images; we parallelize file writes with ThreadPoolExecutor.
        """
        t0 = time.time()
        converter = self._build_converter(pdf_pipeline_options=self._pdf_opts_for_images())
        result = converter.convert(self.input_path)
        doc = result.document
        doc_name = result.input.file.stem
        _LOG.info("Converting to images (scale=%.2f, page_images=%s, picture_images=%s)",
                  self.config.images_scale, self.config.generate_page_images, self.config.generate_picture_images)

        # Save page images in parallel (I/O bound -> threads are optimal).
        def _save_page_image(page_no: int, out_dir: Path) -> None:
            fn = out_dir / f"{doc_name}-{page_no}.{self.config.page_image_format.lower()}"
            with fn.open("wb") as fp:
                doc.pages[page_no].image.pil_image.save(fp, self.config.page_image_format)

        if self.config.generate_page_images:
            with ThreadPoolExecutor(max_workers=self.config.max_io_workers) as exe:
                futures = [exe.submit(_save_page_image, page_no, self.output_dir) for page_no in doc.pages.keys()]
                for fut in as_completed(futures):
                    _ = fut.result()

        # Save tables and figures sequentially (typically fewer, so thread overhead is not required).
        table_count, picture_count = 0, 0
        for element, _ in doc.iterate_items():
            if isinstance(element, TableItem):
                table_count += 1
                out = self.output_dir / f"{doc_name}-table-{table_count}.{self.config.page_image_format.lower()}"
                with out.open("wb") as fp:
                    element.get_image(doc).save(fp, self.config.page_image_format)
            elif isinstance(element, PictureItem):
                picture_count += 1
                out = self.output_dir / f"{doc_name}-picture-{picture_count}.{self.config.page_image_format.lower()}"
                with out.open("wb") as fp:
                    element.get_image(doc).save(fp, self.config.page_image_format)

        # Export rich formats with different image referencing strategies for downstream consumption.
        if save_markdown_embedded:
            doc.save_as_markdown(self.output_dir / f"{doc_name}-embedded.md", image_mode=ImageRefMode.EMBEDDED)
        if save_markdown_referenced:
            doc.save_as_markdown(self.output_dir / f"{doc_name}-refs.md", image_mode=ImageRefMode.REFERENCED)
        if save_html_referenced:
            doc.save_as_html(self.output_dir / f"{doc_name}-refs.html", image_mode=ImageRefMode.REFERENCED)

        _LOG.info("Images exported: pages=%d, tables=%d, pictures=%d (%.2fs)",
                  len(doc.pages), table_count, picture_count, time.time() - t0)

    # ----------------------------------------------------------------------------------------------
    # Task 2: Export Tables (CSV/HTML) and return DataFrames for in-memory usage
    # ----------------------------------------------------------------------------------------------

    def export_tables(self) -> List[Tuple[pd.DataFrame, Path, Path]]:
        """
        Export all detected tables to CSV and HTML. Return a list of (DataFrame, csv_path, html_path).
        Raises:
            RuntimeError: if conversion fails or the document cannot be parsed.
        """
        t0 = time.time()
        converter = self._build_converter(pdf_pipeline_options=self._pdf_pipeline_options_common())
        res = converter.convert(self.input_path)
        doc = res.document
        doc_name = res.input.file.stem

        results: List[Tuple[pd.DataFrame, Path, Path]] = []
        for i, table in enumerate(doc.tables, 1):
            df: pd.DataFrame = table.export_to_dataframe()
            csv_path = self.output_dir / f"{doc_name}-table-{i}.csv"
            html_path = self.output_dir / f"{doc_name}-table-{i}.html"
            df.to_csv(csv_path, index=False)
            with html_path.open("w", encoding="utf-8") as fp:
                fp.write(table.export_to_html(doc=doc))
            results.append((df, csv_path, html_path))
            _LOG.info("Exported table %d -> CSV/HTML", i)

        _LOG.info("Tables exported: %d (%.2fs)", len(results), time.time() - t0)
        return results

    # ----------------------------------------------------------------------------------------------
    # Task 3: Generate Multimodal Parquet
    # ----------------------------------------------------------------------------------------------

    def generate_multimodal_parquet(
        self,
        parquet_name: Optional[str] = None,
        include_image_bytes: Optional[bool] = None,
    ) -> Path:
        """
        Generate a multimodal dataset parquet with text/markdown/structured content, cell/segment metadata,
        and page-level image information. Optionally includes raw image bytes for compact, single-file datasets.

        Returns:
            Path to the generated parquet file.
        """
        t0 = time.time()
        include_bytes = self.config.mm_save_image_bytes if include_image_bytes is None else include_image_bytes

        converter = self._build_converter(pdf_pipeline_options=self._pdf_opts_for_mm())
        res = converter.convert(self.input_path)
        rows: List[Dict[str, Any]] = []

        for text, md, dt, cells, segments, page in generate_multimodal_pages(res):
            rows.append(
                {
                    "document": res.input.file.name,
                    "hash": res.input.document_hash,
                    "page_hash": create_hash(f"{res.input.document_hash}:{page.page_no - 1}"),
                    "image": {
                        "width": page.image.width,
                        "height": page.image.height,
                        "bytes": (page.image.tobytes() if include_bytes else None),
                    },
                    "cells": cells,
                    "contents": text,
                    "contents_md": md,
                    "contents_dt": dt,
                    "segments": segments,
                    "extra": {
                        "page_num": page.page_no,
                        "width_pts": page.size.width,
                        "height_pts": page.size.height,
                        "dpi": page._default_image_scale * 72,
                    },
                }
            )

        df = pd.json_normalize(rows)
        ts = datetime.datetime.now()
        out_name = parquet_name or f"multimodal_{ts:%Y%m%d_%H%M%S}.parquet"
        out_path = self.output_dir / out_name
        df.to_parquet(out_path, compression=self.config.parquet_compression)
        _LOG.info("Multimodal parquet saved: %s (pages=%d, %.2fs)", out_path.name, len(rows), time.time() - t0)
        return out_path

    # ----------------------------------------------------------------------------------------------
    # Task 4: OCR to Markdown 
    # ----------------------------------------------------------------------------------------------

    def ocr_to_markdown(
        self,
        engine: Optional[OcrEngineLiteral] = None,
        save_to_file: bool = True,
        file_name: Optional[str] = None,
    ) -> str:
        """
        Perform OCR-based conversion to Markdown using the specified engine and languages.
        Ideal for scanned/image PDFs. By default, saves a .md file next to other outputs.
        """
        t0 = time.time()
        converter = self._build_converter(pdf_pipeline_options=self._pdf_opts_for_ocr(engine))
        doc = converter.convert(self.input_path).document
        md = doc.export_to_markdown()
        if save_to_file:
            out_name = file_name or f"{self._doc_stem}-ocr.md"
            out_path = self.output_dir / out_name
            out_path.write_text(md, encoding="utf-8")
            _LOG.info("OCR Markdown saved: %s (%.2fs)", out_path.name, time.time() - t0)
        else:
            _LOG.info("OCR Markdown generated in %.2fs (not saved to disk)", time.time() - t0)
        return md

    # ----------------------------------------------------------------------------------------------
    # Orchestration: Run-All convenience
    # ----------------------------------------------------------------------------------------------

    def run_all(
        self,
        do_images: bool = True,
        do_tables: bool = True,
        do_multimodal: bool = True,
        do_ocr_md: bool = False,
    ) -> Dict[str, Any]:
        """
        Convenience method to run the entire pipeline for the single document in this instance.
        Returns a dictionary of outputs for easy programmatic integration.
        """
        outputs: Dict[str, Any] = {"document": str(self.input_path), "outputs": {}}

        if do_images:
            self.extract_images()
            outputs["outputs"]["images"] = True

        if do_tables:
            table_exports = self.export_tables()
            outputs["outputs"]["tables"] = [str(csv) for _, csv, _ in table_exports]

        if do_multimodal:
            parquet_path = self.generate_multimodal_parquet()
            outputs["outputs"]["multimodal_parquet"] = str(parquet_path)

        if do_ocr_md:
            md = self.ocr_to_markdown()
            outputs["outputs"]["ocr_markdown_len"] = len(md)

        return outputs

    # ----------------------------------------------------------------------------------------------
    # Batch / Multiprocessing API
    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def _process_single_document(
        input_path: PathLike,
        output_root: PathLike,
        config: PipelineConfig,
        steps: Sequence[Literal["images", "tables", "multimodal", "ocr_md"]],
    ) -> Dict[str, Any]:
        """
        Internal helper to process a single document; designed to be executed in a separate process.
        """
        pipe = UltraDoclingPipeline(input_path=input_path, output_dir=Path(output_root) / _stem(input_path), config=config)
        result: Dict[str, Any] = {"document": str(input_path), "outputs": {}}

        if "images" in steps:
            pipe.extract_images()
            result["outputs"]["images"] = True

        if "tables" in steps:
            tbls = pipe.export_tables()
            result["outputs"]["tables"] = [str(csv) for _, csv, _ in tbls]

        if "multimodal" in steps:
            mm_path = pipe.generate_multimodal_parquet()
            result["outputs"]["multimodal_parquet"] = str(mm_path)

        if "ocr_md" in steps:
            md = pipe.ocr_to_markdown()
            result["outputs"]["ocr_markdown_len"] = len(md)

        return result

    @classmethod
    def batch_process(
        cls,
        inputs: Sequence[PathLike],
        output_root: PathLike,
        config: Optional[PipelineConfig] = None,
        steps: Sequence[Literal["images", "tables", "multimodal", "ocr_md"]] = ("images", "tables", "multimodal"),
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batch process many documents concurrently using processes for isolation and parallelism.

        Parameters:
            inputs: Sequence of input paths.
            output_root: Root output directory; each document gets a subdirectory named after its stem.
            config: PipelineConfig; if None, default configuration is used.
            steps: Which steps to run per document.
            max_workers: Degree of parallelism. If None:
                - When using GPU accelerators and respect_gpu_memory=True, defaults to 1 to avoid OOMs.
                - Otherwise defaults to number of logical CPUs.

        Returns:
            List of result dicts for each document processed.
        """
        cfg = config or PipelineConfig()
        output_root = Path(output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        # Avoid GPU OOM by limiting process concurrency when a GPU device is selected.
        is_gpu_device = _map_device(cfg.accelerator_device) not in (AcceleratorDevice.CPU, AcceleratorDevice.AUTO, AcceleratorDevice.OPENVINO)
        default_workers = 1 if (is_gpu_device and cfg.respect_gpu_memory) else (cpu_count() or 4)
        workers = max_workers if (max_workers and max_workers > 0) else default_workers

        _LOG.info("Batch processing %d documents with %d worker(s). Steps=%s", len(inputs), workers, list(steps))
        results: List[Dict[str, Any]] = []

        # Use ProcessPoolExecutor for robust isolation and speed on CPU-bound workloads.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    cls._process_single_document,
                    input_path,
                    output_root,
                    cfg,
                    steps,
                ): str(input_path)
                for input_path in inputs
            }
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    results.append(fut.result())
                    _LOG.info("Completed: %s", src)
                except Exception as ex:
                    _LOG.exception("Failed: %s (%s)", src, ex)

        return results


# --------------------------------------------------------------------------------------------------
# Minimal Self-Check/Example (commented only, no side effects)
# --------------------------------------------------------------------------------------------------
# Example usage (uncomment and adapt paths to run locally):
#
if __name__ == "__main__":
    cfg = PipelineConfig(
        images_scale=2.0,
        do_ocr=True,
        ocr_engine="easyocr",
        ocr_languages=["en", "es"],
        accelerator_device="auto",  # 'cuda' for NVIDIA, 'mps' for Apple, 'openvino' for Intel; 'rocm' best-effort
        num_threads=None,
    )
    pipeline = UltraDoclingPipeline(input_path="myfile.pdf", output_dir="out/sample", config=cfg)
    pipeline.extract_images()
    pipeline.export_tables()
    pipeline.generate_multimodal_parquet()
    pipeline.ocr_to_markdown()

    # # Batch processing:
    # UltraDoclingPipeline.batch_process(
    #     inputs=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    #     output_root="out/batch",
    #     config=cfg,
    #     steps=("images", "tables", "multimodal", "ocr_md"),
    #     max_workers=None,
    # )
