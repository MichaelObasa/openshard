from __future__ import annotations

import unittest

from openshard.native.repo_context import (
    NativeRepoContextSummary,
    build_repo_context_summary,
    render_repo_context_summary,
)


class TestNativeRepoContextSummaryDefaults(unittest.TestCase):
    def test_defaults(self):
        s = NativeRepoContextSummary()
        self.assertEqual(s.total_files, 0)
        self.assertEqual(s.top_level_dirs, [])
        self.assertEqual(s.package_files, [])
        self.assertEqual(s.test_markers, [])
        self.assertEqual(s.likely_stack_markers, [])
        self.assertFalse(s.truncated)


class TestBuildRepoContextSummary(unittest.TestCase):

    def test_empty_input(self):
        s = build_repo_context_summary("")
        self.assertEqual(s.total_files, 0)
        self.assertEqual(s.top_level_dirs, [])
        self.assertEqual(s.package_files, [])
        self.assertEqual(s.test_markers, [])
        self.assertEqual(s.likely_stack_markers, [])
        self.assertFalse(s.truncated)

    def test_counts_total_files(self):
        lines = "src/a.py\nsrc/b.py\nREADME.md"
        s = build_repo_context_summary(lines)
        self.assertEqual(s.total_files, 3)

    def test_whitespace_only_lines_ignored(self):
        lines = "src/a.py\n  \n\nsrc/b.py"
        s = build_repo_context_summary(lines)
        self.assertEqual(s.total_files, 2)

    def test_detects_top_level_dirs(self):
        lines = "src/foo.py\nsrc/bar.py\ndocs/index.md"
        s = build_repo_context_summary(lines)
        self.assertEqual(s.top_level_dirs, ["docs", "src"])

    def test_bare_filename_not_a_top_level_dir(self):
        lines = "README.md\nsetup.py"
        s = build_repo_context_summary(lines)
        self.assertEqual(s.top_level_dirs, [])

    def test_detects_package_files(self):
        lines = "pyproject.toml\npackage.json\nsrc/main.py"
        s = build_repo_context_summary(lines)
        self.assertIn("pyproject.toml", s.package_files)
        self.assertIn("package.json", s.package_files)
        self.assertNotIn("main.py", s.package_files)

    def test_detects_all_package_files(self):
        pkgs = [
            "pyproject.toml", "package.json", "pnpm-lock.yaml",
            "yarn.lock", "package-lock.json", "requirements.txt",
            "go.mod", "Cargo.toml", "composer.json", "pom.xml", "build.gradle",
        ]
        s = build_repo_context_summary("\n".join(pkgs))
        for pkg in pkgs:
            self.assertIn(pkg, s.package_files, f"expected {pkg} in package_files")

    def test_detects_test_markers_tests_slash(self):
        lines = "tests/test_auth.py\nsrc/auth.py"
        s = build_repo_context_summary(lines)
        self.assertIn("tests/test_auth.py", s.test_markers)
        self.assertNotIn("src/auth.py", s.test_markers)

    def test_detects_test_markers_test_underscore(self):
        lines = "src/test_utils.py\nsrc/utils.py"
        s = build_repo_context_summary(lines)
        self.assertIn("src/test_utils.py", s.test_markers)

    def test_detects_test_markers_underscore_test_dot(self):
        lines = "src/utils_test.go"
        s = build_repo_context_summary(lines)
        self.assertIn("src/utils_test.go", s.test_markers)

    def test_detects_test_markers_spec(self):
        lines = "src/App.spec.ts\nsrc/App.ts"
        s = build_repo_context_summary(lines)
        self.assertIn("src/App.spec.ts", s.test_markers)
        self.assertNotIn("src/App.ts", s.test_markers)

    def test_detects_test_markers_test_dot(self):
        lines = "src/App.test.tsx"
        s = build_repo_context_summary(lines)
        self.assertIn("src/App.test.tsx", s.test_markers)

    def test_detects_stack_python_extension(self):
        s = build_repo_context_summary("src/main.py")
        self.assertIn("python", s.likely_stack_markers)

    def test_detects_stack_typescript(self):
        s = build_repo_context_summary("src/index.ts\nsrc/App.tsx")
        self.assertIn("typescript", s.likely_stack_markers)
        self.assertEqual(s.likely_stack_markers.count("typescript"), 1)

    def test_detects_stack_javascript(self):
        s = build_repo_context_summary("src/index.js\nsrc/App.jsx")
        self.assertIn("javascript", s.likely_stack_markers)
        self.assertEqual(s.likely_stack_markers.count("javascript"), 1)

    def test_detects_stack_go_extension(self):
        s = build_repo_context_summary("cmd/main.go")
        self.assertIn("go", s.likely_stack_markers)

    def test_detects_stack_rust_extension(self):
        s = build_repo_context_summary("src/main.rs")
        self.assertIn("rust", s.likely_stack_markers)

    def test_detects_stack_java_extension(self):
        s = build_repo_context_summary("src/Main.java")
        self.assertIn("java", s.likely_stack_markers)

    def test_detects_stack_node_from_package_json(self):
        s = build_repo_context_summary("package.json")
        self.assertIn("node", s.likely_stack_markers)

    def test_detects_stack_python_from_pyproject(self):
        s = build_repo_context_summary("pyproject.toml")
        self.assertIn("python", s.likely_stack_markers)

    def test_detects_stack_python_from_requirements(self):
        s = build_repo_context_summary("requirements.txt")
        self.assertIn("python", s.likely_stack_markers)

    def test_detects_stack_rust_from_cargo(self):
        s = build_repo_context_summary("Cargo.toml")
        self.assertIn("rust", s.likely_stack_markers)

    def test_detects_stack_go_from_go_mod(self):
        s = build_repo_context_summary("go.mod")
        self.assertIn("go", s.likely_stack_markers)

    def test_no_duplicate_markers(self):
        lines = "src/a.py\nsrc/b.py\nsrc/c.py"
        s = build_repo_context_summary(lines)
        self.assertEqual(s.likely_stack_markers.count("python"), 1)

    def test_truncates_long_lists_and_sets_flag(self):
        lines = "\n".join(f"pkg{i}/file.py" for i in range(25))
        s = build_repo_context_summary(lines, max_items=20)
        self.assertTrue(s.truncated)
        self.assertLessEqual(len(s.top_level_dirs), 20)

    def test_no_truncation_at_exact_limit(self):
        lines = "\n".join(f"dir{i}/file.py" for i in range(20))
        s = build_repo_context_summary(lines, max_items=20)
        self.assertFalse(s.truncated)
        self.assertEqual(len(s.top_level_dirs), 20)

    def test_lists_are_sorted(self):
        lines = "z_dir/file.py\na_dir/file.go"
        s = build_repo_context_summary(lines)
        self.assertEqual(s.top_level_dirs, sorted(s.top_level_dirs))
        self.assertEqual(s.likely_stack_markers, sorted(s.likely_stack_markers))


class TestRenderRepoContextSummary(unittest.TestCase):

    def _full_summary(self) -> NativeRepoContextSummary:
        return NativeRepoContextSummary(
            total_files=42,
            top_level_dirs=["docs", "src", "tests"],
            package_files=["pyproject.toml"],
            test_markers=["tests/test_main.py"],
            likely_stack_markers=["python"],
            truncated=False,
        )

    def test_render_includes_total_files(self):
        s = self._full_summary()
        rendered = render_repo_context_summary(s)
        self.assertIn("files: 42", rendered)

    def test_render_includes_stack(self):
        s = self._full_summary()
        rendered = render_repo_context_summary(s)
        self.assertIn("python", rendered)

    def test_render_includes_packages(self):
        s = self._full_summary()
        rendered = render_repo_context_summary(s)
        self.assertIn("pyproject.toml", rendered)

    def test_render_includes_dirs(self):
        s = self._full_summary()
        rendered = render_repo_context_summary(s)
        self.assertIn("src", rendered)

    def test_render_includes_test_markers(self):
        s = self._full_summary()
        rendered = render_repo_context_summary(s)
        self.assertIn("tests/test_main.py", rendered)

    def test_render_caps_test_markers_at_five(self):
        s = NativeRepoContextSummary(
            total_files=10,
            test_markers=[f"tests/test_{i}.py" for i in range(10)],
        )
        rendered = render_repo_context_summary(s)
        self.assertIn("tests/test_0.py", rendered)
        self.assertNotIn("tests/test_5.py", rendered)

    def test_render_shows_truncated_flag(self):
        s = NativeRepoContextSummary(total_files=5, truncated=True)
        rendered = render_repo_context_summary(s)
        self.assertIn("(results truncated)", rendered)

    def test_render_bounded_by_limit(self):
        s = self._full_summary()
        rendered = render_repo_context_summary(s, limit=50)
        self.assertLessEqual(len(rendered), 50)

    def test_render_truncation_appends_marker(self):
        s = self._full_summary()
        rendered = render_repo_context_summary(s, limit=50)
        self.assertTrue(rendered.endswith("[truncated]"))

    def test_render_does_not_include_raw_file_list(self):
        lines = "\n".join(f"src/file_{i}.py" for i in range(30))
        s = build_repo_context_summary(lines)
        rendered = render_repo_context_summary(s)
        self.assertNotIn("src/file_0.py", rendered)

    def test_render_empty_summary(self):
        s = NativeRepoContextSummary()
        rendered = render_repo_context_summary(s)
        self.assertIn("[repo context]", rendered)
        self.assertIn("files: 0", rendered)

    def test_render_within_limit_no_truncation_marker(self):
        s = self._full_summary()
        rendered = render_repo_context_summary(s, limit=1200)
        self.assertNotIn("[truncated]", rendered)
