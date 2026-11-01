"""版本发布 checklist 自动检查（R40）。

检查项：
1. pyproject 版本与 deepdossier.__version__ 一致
2. .env.example 覆盖 Settings 全部可配置字段（前缀 DOSSIER_）
3. README 包含当前版本号与测试数量标注
4. CHANGELOG 最新条目包含当前版本号
5. 基准集文件存在且题目 ≥10

用法：python scripts/check_release.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def main() -> None:
    from deepdossier import __version__

    # 1) 版本一致
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
    check("pyproject 版本与 __version__ 一致", m and m.group(1) == __version__, f"pyproject={m and m.group(1)}")

    # 2) .env.example 覆盖全部配置项
    from deepdossier.config import Settings

    fields = set(Settings.model_fields.keys())
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    missing = [f for f in fields if f"DOSSIER_{f.upper()}" not in example]
    check(".env.example 覆盖全部 Settings 字段", not missing, f"缺失: {missing}")

    # 3) README 同步
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check("README 包含当前版本号", __version__ in readme)
    test_count_m = re.search(r"(\d+)\s*项全离线", readme)
    tests_dir = ROOT / "tests"
    actual_tests = len(list(tests_dir.glob("test_*.py")))
    check("README 标注测试文件规模", test_count_m is not None and int(test_count_m.group(1)) >= actual_tests,
          f"README={test_count_m and test_count_m.group(1)}, 实际测试文件 {actual_tests} 个")

    # 4) CHANGELOG
    changelog = ROOT / "CHANGELOG.md"
    check("CHANGELOG 存在且包含当前版本", changelog.exists() and __version__ in changelog.read_text(encoding="utf-8"))

    # 5) 基准集
    bench = ROOT / "benchmarks" / "fact_benchmark.json"
    ok = False
    if bench.exists():
        data = json.loads(bench.read_text(encoding="utf-8"))
        ok = len(data.get("questions", [])) >= 10
    check("事实基准集存在且 ≥10 题", ok)

    print()
    if FAILURES:
        print(f"发布检查未通过（{len(FAILURES)} 项）: {FAILURES}")
        sys.exit(1)
    print(f"发布检查全部通过 ✅ (v{__version__})")


if __name__ == "__main__":
    main()
