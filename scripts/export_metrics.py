import csv
import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = sys.argv[1]
out_csv = sys.argv[2]
files = sorted(glob.glob(f'{log_dir}/**/events.out.tfevents*', recursive=True))
ea = EventAccumulator(files[-1], size_guidance={'scalars': 0})
ea.Reload()
tags = sorted(ea.Tags()['scalars'])
steps = {}
for tag in tags:
    for e in ea.Scalars(tag):
        steps.setdefault(e.step, {'step': e.step})[tag] = e.value
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['step'] + tags)
    writer.writeheader()
    for s in sorted(steps):
        writer.writerow(steps[s])
print('wrote', out_csv, len(steps), 'steps')
