"""只重算保存的训练日志，不加载模型。"""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = json.loads((root / 'reward_log.json').read_text())
assert len(rows) == 40 and [r['step'] for r in rows] == list(range(1, 41))
summary = {'source': 'reward_log.json', 'training_steps': len(rows), 'held_out_evaluation': False}
for field in ['rewards/correctness_reward/mean', 'rewards/format_reward/mean']:
    values = [r[field] for r in rows]
    summary[field] = {'first20_mean': sum(values[:20]) / 20, 'last20_mean': sum(values[20:]) / 20}
assert summary['rewards/correctness_reward/mean'] == {'first20_mean': 0.6, 'last20_mean': 0.95}
print(json.dumps(summary, ensure_ascii=False, indent=2))
