"""
Compute Cohen's kappa between human annotations and LLM judge labels.

Usage on VM:
  cd ~/ircot
  python compute_kappa.py

Requires: human_eval_key.csv (the hidden LLM labels) in the same directory.
"""
import csv
from collections import Counter

# Human annotations (YES = passages contain answer, NO = they don't)
human_labels = {
    0: "NO", 1: "YES", 2: "NO", 3: "NO", 4: "YES",
    5: "NO", 6: "NO", 7: "YES", 8: "YES", 9: "NO",
    10: "NO", 11: "YES", 12: "NO", 13: "YES", 14: "YES",
    15: "NO", 16: "YES", 17: "NO", 18: "NO", 19: "YES",
    20: "NO", 21: "YES", 22: "YES", 23: "YES", 24: "YES",
    25: "YES", 26: "YES", 27: "NO", 28: "NO", 29: "NO",
    30: "NO", 31: "NO", 32: "YES", 33: "YES", 34: "NO",
    35: "YES", 36: "NO", 37: "YES", 38: "NO", 39: "YES",
    40: "YES", 41: "YES", 42: "YES", 43: "NO", 44: "YES",
    45: "YES", 46: "NO", 47: "YES", 48: "YES", 49: "YES",
    50: "YES", 51: "YES", 52: "YES", 53: "NO", 54: "YES",
    55: "YES", 56: "NO", 57: "NO", 58: "YES", 59: "YES",
    60: "NO", 61: "NO", 62: "YES", 63: "NO", 64: "YES",
    65: "YES", 66: "YES", 67: "YES", 68: "YES", 69: "NO",
    70: "NO", 71: "YES", 72: "NO", 73: "YES", 74: "YES",
    75: "YES", 76: "NO", 77: "YES", 78: "NO", 79: "NO",
    80: "YES", 81: "YES", 82: "NO", 83: "NO", 84: "NO",
    85: "NO", 86: "YES", 87: "NO", 88: "NO", 89: "YES",
    90: "NO", 91: "NO", 92: "YES", 93: "NO", 94: "YES",
    95: "NO", 96: "NO", 97: "NO", 98: "YES", 99: "NO",
    100: "YES", 101: "NO", 102: "YES", 103: "YES", 104: "YES",
    105: "NO", 106: "NO", 107: "NO", 108: "YES", 109: "NO",
    110: "YES", 111: "YES", 112: "YES", 113: "YES", 114: "NO",
    115: "YES", 116: "NO", 117: "NO", 118: "NO", 119: "YES",
    120: "NO", 121: "YES", 122: "NO", 123: "YES", 124: "YES",
    125: "NO", 126: "YES", 127: "NO", 128: "NO", 129: "YES",
    130: "NO", 131: "NO", 132: "YES", 133: "YES", 134: "NO",
    135: "NO", 136: "YES", 137: "YES", 138: "YES", 139: "NO",
    140: "NO", 141: "NO", 142: "YES", 143: "YES", 144: "NO",
    145: "NO", 146: "YES", 147: "YES", 148: "NO", 149: "NO",
    150: "YES", 151: "YES", 152: "YES", 153: "YES", 154: "NO",
    155: "YES", 156: "NO", 157: "YES", 158: "YES", 159: "YES",
    160: "YES", 161: "YES", 162: "YES", 163: "YES", 164: "NO",
    165: "NO", 166: "NO", 167: "YES", 168: "NO", 169: "NO",
    170: "YES", 171: "YES", 172: "NO", 173: "YES", 174: "NO",
    175: "YES", 176: "NO", 177: "YES", 178: "YES", 179: "NO",
    180: "NO", 181: "NO", 182: "NO", 183: "YES", 184: "NO",
    185: "YES", 186: "NO", 187: "NO", 188: "YES", 189: "YES",
    190: "NO", 191: "NO", 192: "YES", 193: "YES", 194: "YES",
    195: "YES", 196: "YES", 197: "YES", 198: "NO", 199: "YES",
}

# Load LLM key
llm_labels = {}
with open('human_eval_key.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rid = int(row['id'])
        # Map ANSWERABLE -> YES, NOT-ANSWERABLE -> NO
        llm_labels[rid] = "YES" if row['llm_label'].strip() == "ANSWERABLE" else "NO"

# Align
ids = sorted(human_labels.keys())
human = [human_labels[i] for i in ids]
llm = [llm_labels[i] for i in ids]

# Confusion matrix
tp = sum(1 for h, l in zip(human, llm) if h == "YES" and l == "YES")
tn = sum(1 for h, l in zip(human, llm) if h == "NO" and l == "NO")
fp = sum(1 for h, l in zip(human, llm) if h == "NO" and l == "YES")
fn = sum(1 for h, l in zip(human, llm) if h == "YES" and l == "NO")

n = len(ids)
agreement = (tp + tn) / n
p_yes_h = (tp + fn) / n
p_yes_l = (tp + fp) / n
p_no_h = (tn + fp) / n
p_no_l = (tn + fn) / n
p_e = p_yes_h * p_yes_l + p_no_h * p_no_l
kappa = (agreement - p_e) / (1 - p_e)

print("=" * 50)
print("HUMAN VALIDATION OF LLM JUDGE")
print("=" * 50)
print(f"n = {n}")
print(f"\nHuman: YES={sum(1 for h in human if h=='YES')}, NO={sum(1 for h in human if h=='NO')}")
print(f"LLM:   YES={sum(1 for l in llm if l=='YES')}, NO={sum(1 for l in llm if l=='NO')}")
print(f"\nConfusion Matrix:")
print(f"                LLM YES   LLM NO")
print(f"  Human YES      {tp:4d}      {fn:4d}")
print(f"  Human NO       {fp:4d}      {tn:4d}")
print(f"\nRaw agreement: {agreement:.3f} ({tp+tn}/{n})")
print(f"Expected agreement (chance): {p_e:.3f}")
print(f"Cohen's kappa: {kappa:.3f}")
print()

if kappa >= 0.8:
    interpretation = "almost perfect"
elif kappa >= 0.6:
    interpretation = "substantial"
elif kappa >= 0.4:
    interpretation = "moderate"
else:
    interpretation = "fair or below"
print(f"Interpretation: {interpretation} agreement (Landis & Koch, 1977)")

# Precision/recall of LLM treating human as gold
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print(f"\nLLM vs Human (human as gold):")
print(f"  Precision (YES): {precision:.3f}")
print(f"  Recall (YES):    {recall:.3f}")
print(f"  F1 (YES):        {f1:.3f}")
