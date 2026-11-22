"""Tests for the single ZIP uploaded to the competition portal."""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
import tempfile
import unittest
import warnings
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


VERIFIER = load_module(
    "portal_submission_verifier_under_test",
    "src/07_solve_isling/verify_submission.py",
)
BUILDER = load_module(
    "portal_submission_builder_under_test",
    "src/07_solve_isling/build_submission_bundle.py",
)

REGISTRATION_ID = "YX2600013"
WORK_NAME = "TAILOR-DC-SAFE"
PREFIX = f"能源调度_{REGISTRATION_ID}_{WORK_NAME}"
PORTAL_NAME = f"{PREFIX}_提交材料.zip"
PAPER_NAME = f"{PREFIX}_论文.pdf"
CODE_NAME = f"{PREFIX}_代码.zip"
DATA_NAME = f"{PREFIX}_数据集.zip"
RESULT_NAME = f"{PREFIX}_补充材料.zip"
VIDEO_NAME = f"{PREFIX}_演示视频.mp4"
MANIFEST_NAME = f"{PREFIX}_附件清单.json"


def nested_zip_bytes(label: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="taolin-nested-bytes-") as temporary:
        path = Path(temporary) / "nested.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("payload.txt", label)
        return path.read_bytes()


def component_payloads(*, include_video: bool = False) -> dict[str, bytes]:
    payloads = {
        PAPER_NAME: b"%PDF-1.4\nminimal-test-payload\n%%EOF\n",
        CODE_NAME: nested_zip_bytes("code"),
        DATA_NAME: nested_zip_bytes("data"),
        RESULT_NAME: nested_zip_bytes("results"),
        MANIFEST_NAME: json.dumps(
            {
                "registration_id": REGISTRATION_ID,
                "work_name": WORK_NAME,
                "video_required": False,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
    }
    if include_video:
        payloads[VIDEO_NAME] = b"\x00\x00\x00\x18ftypisom" + b"video"
    return payloads


def write_outer_zip(
    path: Path,
    payloads: dict[str, bytes],
    *,
    order: list[str] | None = None,
    timestamp: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0),
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in order or sorted(payloads):
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(info, payloads[name])


def verifier_args(*, require_video: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        require_video=require_video,
        max_scan_bytes=32 * 1024 * 1024,
        max_archive_bytes=1024 * 1024 * 1024,
        max_members=10_000,
        max_member_bytes=128 * 1024 * 1024,
        max_uncompressed_bytes=512 * 1024 * 1024,
        max_member_compression_ratio=500.0,
        max_total_compression_ratio=100.0,
    )


class PortalSubmissionZipTests(unittest.TestCase):
    def test_no_video_portal_zip_is_accepted_and_safely_reuses_component_verifier(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-portal-ok-") as temporary:
            archive_path = Path(temporary) / PORTAL_NAME
            write_outer_zip(archive_path, component_payloads())
            recorder = VERIFIER.Recorder()

            def fake_component_verifier(staging_dir, root, nested_recorder, args):
                self.assertEqual(
                    {path.name for path in staging_dir.iterdir()},
                    set(component_payloads()),
                )
                for check_id in VERIFIER.STAGING_MANDATORY_CHECKS:
                    nested_recorder.passed(check_id, "component check passed")

            with mock.patch.object(
                VERIFIER,
                "verify_staging_bundle",
                side_effect=fake_component_verifier,
            ) as component_verifier:
                VERIFIER.verify_portal_zip(
                    archive_path,
                    ROOT,
                    recorder,
                    verifier_args(),
                )

            component_verifier.assert_called_once()
            self.assertTrue(recorder.ready_for(VERIFIER.PORTAL_MANDATORY_CHECKS))

    def test_require_video_rejects_no_video_portal_zip(self) -> None:
        with self.assertRaises(BUILDER.SubmissionError):
            BUILDER.validate_video(None, required=True)
        with tempfile.TemporaryDirectory(prefix="taolin-portal-video-") as temporary:
            archive_path = Path(temporary) / PORTAL_NAME
            write_outer_zip(archive_path, component_payloads())
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(
                    archive_path,
                    require_video=True,
                )

    def test_optional_video_portal_zip_has_six_members(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-portal-with-video-") as temporary:
            archive_path = Path(temporary) / PORTAL_NAME
            write_outer_zip(archive_path, component_payloads(include_video=True))
            state = VERIFIER.inspect_portal_zip(archive_path, require_video=True)
            self.assertTrue(state["video_present"])
            self.assertEqual(state["member_count"], 6)

    def test_portal_size_limit_is_strictly_less_than_50_mb(self) -> None:
        BUILDER.validate_portal_archive_byte_count(49_999_999)
        with self.assertRaises(BUILDER.SubmissionError):
            BUILDER.validate_portal_archive_byte_count(50_000_000)
        with self.assertRaises(BUILDER.SubmissionError):
            BUILDER.validate_portal_archive_byte_count(50_000_001)

    def test_missing_extra_renamed_and_directory_members_are_rejected(self) -> None:
        valid = component_payloads()
        cases = {
            "missing": {name: value for name, value in valid.items() if name != PAPER_NAME},
            "extra": {**valid, "unexpected.txt": b"extra"},
            "renamed": {
                ("renamed.pdf" if name == PAPER_NAME else name): value
                for name, value in valid.items()
            },
            "directory": {**valid, "folder/payload.txt": b"nested"},
        }
        with tempfile.TemporaryDirectory(prefix="taolin-portal-members-") as temporary:
            for label, payloads in cases.items():
                with self.subTest(label=label):
                    archive_path = Path(temporary) / PORTAL_NAME
                    write_outer_zip(archive_path, payloads)
                    with self.assertRaises(ValueError):
                        VERIFIER.inspect_portal_zip(archive_path, require_video=False)

    def test_wrong_archive_name_order_timestamp_and_case_collision_are_rejected(self) -> None:
        valid = component_payloads()
        with tempfile.TemporaryDirectory(prefix="taolin-portal-metadata-") as temporary:
            root = Path(temporary)
            wrong_name = root / "wrong.zip"
            write_outer_zip(wrong_name, valid)
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(wrong_name, require_video=False)

            wrong_order = root / PORTAL_NAME
            write_outer_zip(wrong_order, valid, order=list(reversed(sorted(valid))))
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(wrong_order, require_video=False)

            write_outer_zip(wrong_order, valid, timestamp=(2026, 7, 23, 0, 0, 0))
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(wrong_order, require_video=False)

            collision = dict(valid)
            collision[PAPER_NAME.upper()] = b"collision"
            write_outer_zip(wrong_order, collision)
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(wrong_order, require_video=False)

    def test_path_traversal_duplicate_and_unknown_nested_archive_are_rejected(self) -> None:
        valid = component_payloads()
        with tempfile.TemporaryDirectory(prefix="taolin-portal-hostile-") as temporary:
            root = Path(temporary)
            archive_path = root / PORTAL_NAME

            traversal = dict(valid)
            traversal["../escape.txt"] = b"escape"
            write_outer_zip(archive_path, traversal)
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(archive_path, require_video=False)

            polyglot = dict(valid)
            polyglot[PAPER_NAME] = valid[PAPER_NAME] + nested_zip_bytes("appended archive")
            write_outer_zip(archive_path, polyglot)
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(archive_path, require_video=False)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(archive_path, "w") as archive:
                    for name, payload in sorted(valid.items()):
                        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                        archive.writestr(info, payload)
                        if name == PAPER_NAME:
                            archive.writestr(info, payload)
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(archive_path, require_video=False)

            nested = dict(valid)
            nested[PAPER_NAME] = nested_zip_bytes("hidden archive")
            write_outer_zip(archive_path, nested)
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(archive_path, require_video=False)

    def test_crc_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-portal-tamper-") as temporary:
            archive_path = Path(temporary) / PORTAL_NAME
            valid = component_payloads()
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                for name, payload in sorted(valid.items()):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED
                    archive.writestr(info, payload)
            with zipfile.ZipFile(archive_path, "r") as archive:
                info = archive.getinfo(PAPER_NAME)
                payload_offset = info.header_offset + 30 + len(info.filename.encode("utf-8"))
            with archive_path.open("r+b") as handle:
                handle.seek(payload_offset)
                original = handle.read(1)
                handle.seek(payload_offset)
                handle.write(bytes([original[0] ^ 0xFF]))
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(archive_path, require_video=False)

    def test_encrypted_member_flag_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-portal-encrypted-") as temporary:
            archive_path = Path(temporary) / PORTAL_NAME
            write_outer_zip(archive_path, component_payloads())
            payload = bytearray(archive_path.read_bytes())
            central_header = payload.find(b"PK\x01\x02")
            self.assertGreaterEqual(central_header, 0)
            flag_offset = central_header + 8
            flags = int.from_bytes(payload[flag_offset : flag_offset + 2], "little")
            payload[flag_offset : flag_offset + 2] = (flags | 0x1).to_bytes(2, "little")
            archive_path.write_bytes(payload)
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(archive_path, require_video=False)

    def test_symlink_and_special_outer_members_are_rejected(self) -> None:
        valid = component_payloads()
        with tempfile.TemporaryDirectory(prefix="taolin-portal-filetype-") as temporary:
            archive_path = Path(temporary) / PORTAL_NAME
            for label, mode in (
                ("symlink", stat.S_IFLNK | 0o777),
                ("fifo", stat.S_IFIFO | 0o644),
            ):
                with self.subTest(label=label):
                    with zipfile.ZipFile(archive_path, "w") as archive:
                        for name, payload in sorted(valid.items()):
                            info = zipfile.ZipInfo(
                                name,
                                date_time=(1980, 1, 1, 0, 0, 0),
                            )
                            info.create_system = 3
                            info.compress_type = zipfile.ZIP_DEFLATED
                            info.external_attr = (
                                mode if name == MANIFEST_NAME else stat.S_IFREG | 0o644
                            ) << 16
                            archive.writestr(info, payload)
                    with self.assertRaises(ValueError):
                        VERIFIER.inspect_portal_zip(archive_path, require_video=False)

    def test_outer_zip_honors_configured_expansion_and_ratio_limits(self) -> None:
        payloads = component_payloads()
        payloads[PAPER_NAME] = b"%PDF-1.4\n" + b"0" * 16_384 + b"\n%%EOF\n"
        with tempfile.TemporaryDirectory(prefix="taolin-portal-limits-") as temporary:
            archive_path = Path(temporary) / PORTAL_NAME
            write_outer_zip(archive_path, payloads)
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(
                    archive_path,
                    require_video=False,
                    max_member_compression_ratio=2.0,
                )
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(
                    archive_path,
                    require_video=False,
                    max_member_bytes=1_024,
                )

    def test_builder_outer_zip_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-portal-determinism-") as temporary:
            root = Path(temporary)
            components = root / "components"
            components.mkdir()
            for name, payload in component_payloads().items():
                (components / name).write_bytes(payload)
            first = root / "first.zip"
            second = root / "second.zip"
            BUILDER.write_portal_zip(first, components, sorted(component_payloads()))
            BUILDER.write_portal_zip(second, components, sorted(component_payloads()))
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_publish_directory_must_contain_only_the_portal_zip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-portal-publish-") as temporary:
            root = Path(temporary)
            archive_path = root / PORTAL_NAME
            archive_path.write_bytes(b"zip")
            BUILDER.verify_portal_publish_directory(root, PORTAL_NAME)
            (root / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaises(BUILDER.SubmissionError):
                BUILDER.verify_portal_publish_directory(root, PORTAL_NAME)

    def test_verifier_rejects_multiple_publish_artifacts_next_to_portal_zip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="taolin-portal-unique-") as temporary:
            root = Path(temporary)
            archive_path = root / PORTAL_NAME
            write_outer_zip(archive_path, component_payloads())
            (root / "second-package.zip").write_bytes(archive_path.read_bytes())
            with self.assertRaises(ValueError):
                VERIFIER.inspect_portal_zip(archive_path, require_video=False)


if __name__ == "__main__":
    unittest.main()
