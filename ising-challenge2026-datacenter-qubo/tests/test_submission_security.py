"""Regression tests for the portal-facing submission security gates."""

from __future__ import annotations

import importlib.util
import csv
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = load_module("submission_verifier_under_test", "src/07_solve_isling/verify_submission.py")
BUILDER = load_module("submission_builder_under_test", "src/07_solve_isling/build_submission_bundle.py")
FINAL_BUILDER = load_module("final_builder_under_test", "src/07_solve_isling/build_final_deliverable.py")


def mp4_box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


class SubmissionSecurityTests(unittest.TestCase):
    def run_tampered_embedded_validator(
        self,
        filename: str,
        column: str,
        value: str,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        source = ROOT / "deliverables" / "final_solution_package"
        if not source.is_dir():
            self.skipTest("Canonical final solution package is unavailable")
        with tempfile.TemporaryDirectory(prefix="taolin-validator-tamper-") as temporary:
            package = Path(temporary) / "final_solution_package"
            shutil.copytree(source, package)
            (package / "validate_final_solution.py").write_text(
                FINAL_BUILDER.VALIDATOR_CODE,
                encoding="utf-8",
            )
            csv_path = package / filename
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
            rows[0][column] = value
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            process = subprocess.run(
                [sys.executable, "-I", "-B", "validate_final_solution.py"],
                cwd=package,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            payload = json.loads(
                (package / "VALIDATION_RECOMPUTED.json").read_text(encoding="utf-8")
            )
            return process, payload

    def test_missing_or_outdated_verifier_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-no-verifier-") as temporary:
            with mock.patch.object(BUILDER, "ROOT", Path(temporary)):
                with self.assertRaises(BUILDER.SubmissionError):
                    BUILDER.run_prebuild_verifier_if_available()
        with self.assertRaises(BUILDER.SubmissionError):
            BUILDER.run_staging_verifier_if_supported(
                None,
                "",
                ROOT,
                require_video=False,
            )
        with self.assertRaises(BUILDER.SubmissionError):
            BUILDER.run_staging_verifier_if_supported(
                ROOT / "src/07_solve_isling/verify_submission.py",
                "usage without the required staging flag",
                ROOT,
                require_video=False,
            )

    def test_transient_windows_lock_is_retried(self) -> None:
        with mock.patch.object(
            BUILDER.os,
            "replace",
            side_effect=[PermissionError("locked"), None],
        ) as replace_mock, mock.patch.object(BUILDER.time, "sleep") as sleep_mock:
            BUILDER.replace_with_retry(Path("source"), Path("destination"))
        self.assertEqual(replace_mock.call_count, 2)
        sleep_mock.assert_called_once()

    def test_windows_unsafe_archive_names_are_rejected(self) -> None:
        for name in ("payload.bin:credential", "CON.txt", "folder./file.txt", "name. "):
            with self.subTest(name=name):
                self.assertIsNotNone(VERIFIER.validate_relative_path(name))
                with self.assertRaises(BUILDER.SubmissionError):
                    BUILDER._validate_archive_name(name)

    def test_generic_json_secret_in_lock_file_is_detected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-secret-test-") as temporary:
            root = Path(temporary)
            sensitive_key = "api_" + "key"
            synthetic_value = "generic-" + "secret-value"
            (root / "credentials.lock").write_text(
                json.dumps({sensitive_key: synthetic_value}),
                encoding="utf-8",
            )
            recorder = VERIFIER.Recorder()
            VERIFIER.scan_package(root, recorder, 1024 * 1024)
            secret_check = next(
                check for check in recorder.checks if check.check_id == "package_secret_scan"
            )
            self.assertEqual(secret_check.status, "fail")

    def test_expansion_failure_skips_crc_decompression(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-bomb-test-") as temporary:
            root = Path(temporary)
            archive_path = root / "high-ratio.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("payload.bin", b"0" * (1024 * 1024))
            args = SimpleNamespace(
                max_archive_bytes=10 * 1024 * 1024,
                max_members=100,
                max_member_bytes=2 * 1024 * 1024,
                max_uncompressed_bytes=2 * 1024 * 1024,
                max_member_compression_ratio=2.0,
                max_total_compression_ratio=2.0,
            )
            recorder = VERIFIER.Recorder()
            with mock.patch.object(
                zipfile.ZipFile,
                "testzip",
                side_effect=AssertionError("CRC decompression must be skipped"),
            ):
                state = VERIFIER.verify_zip(
                    archive_path,
                    root,
                    None,
                    recorder,
                    args,
                    check_payload_consistency=False,
                )
            self.assertIsNotNone(state)
            self.assertFalse(state["safe"])
            safety = next(check for check in recorder.checks if check.check_id == "zip_safety")
            self.assertTrue(safety.evidence["crc_test_skipped"])

    def test_renamed_nested_zip_is_detected_by_signature(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-nested-test-") as temporary:
            path = Path(temporary) / "payload.bin"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("inside.txt", "content")
            self.assertEqual(VERIFIER.nested_archive_kind(path), "signature:zip")

    def test_header_only_mp4_with_no_samples_is_rejected(self) -> None:
        movie_header = b"\x00\x00\x00\x00" + b"\x00" * 8 + struct.pack(">II", 1000, 1000)
        handler = b"\x00\x00\x00\x00" + b"\x00" * 4 + b"vide"
        payload = b"".join(
            (
                mp4_box(b"ftyp", b"isom\x00\x00\x02\x00isom"),
                mp4_box(
                    b"moov",
                    mp4_box(b"mvhd", movie_header)
                    + mp4_box(b"trak", mp4_box(b"mdia", mp4_box(b"hdlr", handler))),
                ),
                mp4_box(b"mdat", b"\x00" * 2048),
            )
        )
        with tempfile.TemporaryDirectory(prefix="taolin-video-test-") as temporary:
            path = Path(temporary) / "header-only.mp4"
            path.write_bytes(payload)
            with self.assertRaises(BUILDER.SubmissionError):
                BUILDER.validate_video(path, required=True)

    def test_pdf_with_javascript_is_rejected(self) -> None:
        try:
            try:
                from pypdf import PdfWriter
            except ImportError:
                from PyPDF2 import PdfWriter
        except ImportError:
            self.skipTest("No PDF writer is available")
        with tempfile.TemporaryDirectory(prefix="taolin-pdf-test-") as temporary:
            path = Path(temporary) / "active.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            writer.add_js("app.alert('test')")
            writer.add_metadata({"/Subject": "x" * 2048})
            with path.open("wb") as handle:
                writer.write(handle)
            with self.assertRaises(BUILDER.SubmissionError):
                BUILDER.validate_paper(path)

    def test_pdf_with_acroform_javascript_is_rejected(self) -> None:
        try:
            from pypdf import PdfWriter
            from pypdf.generic import ArrayObject, DictionaryObject, NameObject, TextStringObject
        except ImportError:
            self.skipTest("pypdf is required for the AcroForm security probe")
        with tempfile.TemporaryDirectory(prefix="taolin-acroform-test-") as temporary:
            path = Path(temporary) / "active-form.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            action = DictionaryObject(
                {
                    NameObject("/S"): NameObject("/JavaScript"),
                    NameObject("/JS"): TextStringObject("app.alert('field action')"),
                }
            )
            field = DictionaryObject(
                {
                    NameObject("/FT"): NameObject("/Tx"),
                    NameObject("/T"): TextStringObject("field"),
                    NameObject("/AA"): DictionaryObject({NameObject("/K"): action}),
                }
            )
            field_reference = writer._add_object(field)
            writer._root_object[NameObject("/AcroForm")] = DictionaryObject(
                {NameObject("/Fields"): ArrayObject([field_reference])}
            )
            writer.add_metadata({"/Subject": "x" * 4096})
            with path.open("wb") as handle:
                writer.write(handle)
            with self.assertRaises(BUILDER.SubmissionError):
                BUILDER.validate_paper(path)

    def test_pdf_with_page_associated_file_is_rejected(self) -> None:
        try:
            from pypdf import PdfWriter
            from pypdf.generic import (
                ArrayObject,
                DecodedStreamObject,
                DictionaryObject,
                NameObject,
                TextStringObject,
            )
        except ImportError:
            self.skipTest("pypdf is required for the associated-file security probe")
        with tempfile.TemporaryDirectory(prefix="taolin-page-af-test-") as temporary:
            path = Path(temporary) / "page-associated-file.pdf"
            writer = PdfWriter()
            page = writer.add_blank_page(width=612, height=792)
            embedded = DecodedStreamObject()
            embedded.set_data(b"embedded payload")
            embedded[NameObject("/Type")] = NameObject("/EmbeddedFile")
            embedded_reference = writer._add_object(embedded)
            filespec = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Filespec"),
                    NameObject("/F"): TextStringObject("payload.bin"),
                    NameObject("/EF"): DictionaryObject(
                        {NameObject("/F"): embedded_reference}
                    ),
                }
            )
            page[NameObject("/AF")] = ArrayObject([writer._add_object(filespec)])
            writer.add_metadata({"/Subject": "x" * 4096})
            with path.open("wb") as handle:
                writer.write(handle)
            with self.assertRaises(BUILDER.SubmissionError):
                BUILDER.validate_paper(path)

    def test_invalid_mp4_sample_tables_are_rejected(self) -> None:
        ftyp = mp4_box(b"ftyp", b"isom\x00\x00\x02\x00isom")
        movie_header = b"\x00\x00\x00\x00" + b"\x00" * 8 + struct.pack(">II", 1000, 1000)
        handler = b"\x00" * 8 + b"vide"

        def movie(
            chunk_offset: int,
            *,
            declared_descriptions: int,
            sample_count: int,
            sample_to_chunk: tuple[int, int, int],
            timing: tuple[int, int],
        ) -> bytes:
            sample_entry = struct.pack(">I4s", 86, b"avc1") + b"\x00" * 78
            stsd = mp4_box(
                b"stsd",
                b"\x00" * 4 + struct.pack(">I", declared_descriptions) + sample_entry,
            )
            stsz = mp4_box(b"stsz", b"\x00" * 4 + struct.pack(">II", 1024, sample_count))
            stsc = mp4_box(
                b"stsc",
                b"\x00" * 4 + struct.pack(">I", 1) + struct.pack(">III", *sample_to_chunk),
            )
            stts = mp4_box(
                b"stts",
                b"\x00" * 4 + struct.pack(">I", 1) + struct.pack(">II", *timing),
            )
            stco = mp4_box(
                b"stco",
                b"\x00" * 4 + struct.pack(">I", 1) + struct.pack(">I", chunk_offset),
            )
            stbl = mp4_box(b"stbl", stsd + stsz + stsc + stts + stco)
            minf = mp4_box(b"minf", stbl)
            mdia = mp4_box(b"mdia", mp4_box(b"hdlr", handler) + minf)
            return mp4_box(b"moov", mp4_box(b"mvhd", movie_header) + mp4_box(b"trak", mdia))

        with tempfile.TemporaryDirectory(prefix="taolin-invalid-stbl-") as temporary:
            cases = {
                "zero_entries": {
                    "declared_descriptions": 1,
                    "sample_count": 1,
                    "sample_to_chunk": (0, 0, 0),
                    "timing": (1, 0),
                },
                "missing_description": {
                    "declared_descriptions": 2,
                    "sample_count": 1,
                    "sample_to_chunk": (1, 1, 2),
                    "timing": (1, 1),
                },
                "excessive_sample_count": {
                    "declared_descriptions": 1,
                    "sample_count": 0xFFFFFFFF,
                    "sample_to_chunk": (1, 0xFFFFFFFF, 1),
                    "timing": (0xFFFFFFFF, 1),
                },
            }
            for name, kwargs in cases.items():
                with self.subTest(name=name):
                    provisional_movie = movie(0, **kwargs)
                    mdat_payload_offset = len(ftyp) + len(provisional_movie) + 8
                    payload = (
                        ftyp
                        + movie(mdat_payload_offset, **kwargs)
                        + mp4_box(b"mdat", b"\x00" * 2048)
                    )
                    path = Path(temporary) / f"{name}.mp4"
                    path.write_bytes(payload)
                    with self.assertRaises(BUILDER.SubmissionError):
                        BUILDER.validate_video(path, required=True)

    def test_selected_row_must_match_canonical_action(self) -> None:
        action = {
            "action_id": "act_1",
            "job_id": "job_1",
            "site": "site_1",
            "start_slot": 1,
            "end_slot": 2,
            "mode": "normal",
            "gpu_req": 1,
            "power_mw": 0.1,
            "energy_mwh": 0.1,
            "energy_cost_usd": 1.0,
            "sla_cost_usd": 0.0,
            "extra_mode_cost_usd": 0.0,
            "action_cost": 1.0,
            "is_dummy": False,
            "gpu_vector": [0.0, 1.0],
            "power_vector_mw": [0.0, 0.1],
            "conflict_keys": ["job:job_1"],
            "source": "unit_test",
        }
        actions = FINAL_BUILDER.pd.DataFrame([action])
        selected = actions.copy()
        selected["selection_source"] = "accepted"
        selected.loc[0, "action_cost"] = 9.0
        result = {
            "selected_job_count": 1,
            "selected_action_count": 1,
            "risk_accepted_solution": {
                "source": "accepted",
                "metrics": {
                    "dummy_actions": 0,
                    "one_hot": {"total": 0},
                },
            },
        }
        validation = FINAL_BUILDER.validate_solution(result, selected, actions)
        self.assertEqual(validation["status"], "fail")
        self.assertFalse(validation["checks"]["selected_rows_match_canonical_actions"])

    def test_compact_solution_and_boolean_values_fail_closed(self) -> None:
        full = {
            "job_id": "job_1",
            "action_id": "act_1",
            "site": "site_1",
            "start_slot": "1",
            "end_slot": "2",
            "mode": "normal",
            "gpu_req": "1",
            "power_mw": "0.1",
            "energy_mwh": "0.1",
            "action_cost": "1.0",
            "is_dummy": "False",
        }
        compact = dict(full)
        compact["site"] = "TAMPERED_SITE"
        self.assertFalse(VERIFIER.csv_compact_rows_match(compact, full))
        with self.assertRaises(ValueError):
            VERIFIER.parse_bool("MALFORMED_FALSE_VALUE")
        with self.assertRaises(ValueError):
            FINAL_BUILDER.strict_bool_value("MALFORMED_FALSE_VALUE")

    def test_embedded_validator_rejects_tampered_compact_solution(self) -> None:
        process, payload = self.run_tampered_embedded_validator(
            "FINAL_SOLUTION.csv",
            "site",
            "TAMPERED_SITE",
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(payload["status"], "fail")
        self.assertFalse(payload["checks"]["final_solution_matches_full_selected_actions"])

    def test_embedded_validator_rejects_malformed_boolean(self) -> None:
        process, payload = self.run_tampered_embedded_validator(
            "FULL_SELECTED_ACTIONS.csv",
            "is_dummy",
            "MALFORMED_FALSE_VALUE",
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(payload["status"], "fail")
        self.assertFalse(payload["checks"]["boolean_values_valid"])

    def test_embedded_validator_rejects_tampered_selection_source(self) -> None:
        process, payload = self.run_tampered_embedded_validator(
            "FULL_SELECTED_ACTIONS.csv",
            "selection_source",
            "TAMPERED_PROVENANCE",
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(payload["status"], "fail")
        self.assertFalse(payload["checks"]["selection_source_matches_metrics"])

    def test_embedded_validator_rejects_duplicate_csv_header(self) -> None:
        source = ROOT / "deliverables" / "final_solution_package"
        if not source.is_dir():
            self.skipTest("Canonical final solution package is unavailable")
        with tempfile.TemporaryDirectory(prefix="taolin-duplicate-header-") as temporary:
            package = Path(temporary) / "final_solution_package"
            shutil.copytree(source, package)
            (package / "validate_final_solution.py").write_text(
                FINAL_BUILDER.VALIDATOR_CODE,
                encoding="utf-8",
            )
            csv_path = package / "FINAL_SOLUTION.csv"
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            site_index = rows[0].index("site")
            rows[0].append("site")
            for row_number, row in enumerate(rows[1:], start=1):
                canonical_site = row[site_index]
                if row_number == 1:
                    row[site_index] = "TAMPERED_FIRST_SITE_COLUMN"
                row.append(canonical_site)
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                csv.writer(handle).writerows(rows)
            process = subprocess.run(
                [sys.executable, "-I", "-B", "validate_final_solution.py"],
                cwd=package,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("duplicate CSV header", process.stderr + process.stdout)

    def test_final_builder_rejects_linked_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-link-target-") as outside:
            outside_file = Path(outside) / "outside.txt"
            outside_file.write_text("outside", encoding="utf-8")
            with tempfile.TemporaryDirectory(dir=ROOT / "temp", prefix="link-source-") as inside:
                link = Path(inside) / "linked.txt"
                try:
                    link.symlink_to(outside_file)
                except OSError as error:
                    self.skipTest(f"Symlink creation unavailable: {error}")
                with self.assertRaises(RuntimeError):
                    FINAL_BUILDER.validate_source_file(link, label="test source")
                with self.assertRaises(ValueError):
                    VERIFIER.validate_unlinked_path_components(link, label="test source")

    def test_linked_ancestor_is_rejected(self) -> None:
        temp_root = ROOT / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root, prefix="linked-ancestor-") as temporary:
            parent = Path(temporary)
            real = parent / "real"
            real.mkdir()
            payload = real / "payload.txt"
            payload.write_text("payload", encoding="utf-8")
            linked = parent / "linked"
            try:
                linked.symlink_to(real, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Directory symlink creation unavailable: {error}")
            linked_payload = linked / "payload.txt"
            with self.assertRaises(BUILDER.SubmissionError):
                BUILDER.validate_repository_source(
                    linked_payload,
                    relative=linked_payload.relative_to(ROOT).as_posix(),
                    label="test source",
                )
            with self.assertRaises(RuntimeError):
                FINAL_BUILDER.validate_source_file(linked_payload, label="test source")
            with self.assertRaises(ValueError):
                VERIFIER.validate_unlinked_path_components(linked_payload, label="test source")

    def test_ready_requires_exact_all_pass_mandatory_checks(self) -> None:
        recorder = VERIFIER.Recorder()
        for check_id in sorted(VERIFIER.CANONICAL_MANDATORY_CHECKS):
            recorder.passed(check_id, "pass")
        self.assertTrue(recorder.ready_for(VERIFIER.CANONICAL_MANDATORY_CHECKS))
        recorder.checks[-1].status = "skip"
        self.assertFalse(recorder.ready)
        self.assertFalse(recorder.ready_for(VERIFIER.CANONICAL_MANDATORY_CHECKS))
        recorder.checks[-1].status = "pass"
        recorder.checks.pop()
        self.assertFalse(recorder.ready_for(VERIFIER.CANONICAL_MANDATORY_CHECKS))


if __name__ == "__main__":
    unittest.main()
