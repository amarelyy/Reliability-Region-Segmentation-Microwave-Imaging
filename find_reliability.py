from pathlib import Path
import re

pattern = re.compile(r'(peak_dominance|boundary_risk|reliability_score|evaluate_scan_reliability)', re.IGNORECASE)

for py in Path('.').rglob('*.py'):
    if '.venv' in str(py) or 'venv' in str(py):
        continue
    try:
        text = py.read_text(encoding='utf-8')
        matches = pattern.findall(text)
        if matches:
            print(f"{py}: {set(matches)}")
    except:
        pass