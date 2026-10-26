"""YOLO26 vehicle tracking: persistent IDs, trails, MP4 and CSV output."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

from tracking import TrackHistory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/vehicles.mp4', help='Video file or camera index, e.g. 0')
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--tracker', default='botsort.yaml', help='Built-in tracker name or custom YAML path')
    parser.add_argument('--output', type=Path, default=Path('outputs') / datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    parser.add_argument('--device', default='auto', help='auto, cpu, or GPU index such as 0')
    parser.add_argument('--classes', type=int, nargs='+', default=[2, 3, 5, 7], help='COCO IDs; default car/motorcycle/bus/truck')
    parser.add_argument('--all-classes', action='store_true')
    parser.add_argument('--conf', type=float, default=0.1)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--width', type=int, default=1280, help='Maximum output width; 0 keeps original resolution')
    parser.add_argument('--trail-length', type=int, default=60)
    parser.add_argument('--max-age', type=int, default=30)
    parser.add_argument('--max-frames', type=int, default=0, help='0 processes the whole video')
    parser.add_argument('--show', action='store_true', help='Show a live window; Q or Esc stops')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or args.imgsz < 32 or args.max_frames < 0 or args.width < 0 or args.width == 1:
        parser.error('conf must be in [0,1], imgsz >= 32, max-frames >= 0, width is 0 or >= 2')
    if args.trail_length < 1 or args.max_age < 0:
        parser.error('trail-length must be positive, max-age nonnegative')
    if not args.source.isdecimal() and not Path(args.source).is_file():
        parser.error(f'Source does not exist: {args.source}')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}. Choose a new directory.')
    return args


def run(args):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    model = YOLO(args.model)
    capture = cv2.VideoCapture(int(args.source) if args.source.isdecimal() else args.source)
    writer = None
    frames = 0
    seen = set()
    history = TrackHistory(args.trail_length, args.max_age)
    started = time.perf_counter()
    stopped_early = False
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f'Cannot decode source: {args.source}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        height, width = frame.shape[:2]
        if args.width and width > args.width:
            height = max(2, round(height * args.width / width) // 2 * 2)
            width = args.width // 2 * 2
        # MP4 encoders require even dimensions.
        width, height = width // 2 * 2, height // 2 * 2
        args.output.mkdir(parents=True, exist_ok=False)
        writer = cv2.VideoWriter(str(args.output / 'tracked.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Cannot open MP4 video writer')
        with (args.output / 'tracks.csv').open('w', newline='', encoding='utf-8') as csv_file:
            table = csv.writer(csv_file)
            table.writerow(['frame', 'time_seconds', 'track_id', 'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
            while ok:
                frames += 1
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                # persist=True keeps tracker state across consecutive frames.
                result = model.track(frame, persist=True, tracker=args.tracker, device=device,
                                     classes=None if args.all_classes else args.classes,
                                     conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
                canvas = frame.copy()
                centers = {}
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    for xyxy, ident, cls, confidence in zip(boxes.xyxy.cpu().tolist(), boxes.id.int().cpu().tolist(),
                                                          boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist()):
                        x1, y1, x2, y2 = map(int, xyxy)
                        centers[ident] = ((x1 + x2) // 2, (y1 + y2) // 2)
                        seen.add(ident)
                        color = ((ident * 67) % 180 + 75, (ident * 37) % 180 + 75, (ident * 97) % 180 + 75)
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                        label = f'ID {ident} | {model.names[cls]} {confidence:.2f}'
                        cv2.putText(canvas, label, (max(0, x1), max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        table.writerow([frames, round((frames - 1) / fps, 4), ident, cls, model.names[cls],
                                        round(confidence, 5), *[round(x, 2) for x in xyxy]])
                # Empty frames still expire lost trajectories to bound memory.
                history.update(frames, centers)
                for ident in centers:
                    points = np.array(history.points[ident], dtype=np.int32).reshape(-1, 1, 2)
                    color = ((ident * 67) % 180 + 75, (ident * 37) % 180 + 75, (ident * 97) % 180 + 75)
                    cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
                elapsed = time.perf_counter() - started
                cv2.rectangle(canvas, (0, 0), (min(width, 900), 38), (25, 25, 25), -1)
                cv2.putText(canvas, f'YOLO26 | {Path(args.tracker).stem} | frame {frames} | active {len(centers)} | {frames / elapsed:.1f} FPS',
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                writer.write(canvas)
                if frames == 1 or frames % 100 == 0:
                    cv2.imwrite(str(args.output / 'preview.jpg'), canvas)
                    print(f'Frames: {frames}, active tracks: {len(centers)}, speed: {frames / elapsed:.1f} FPS', flush=True)
                if args.show:
                    cv2.imshow('YOLO26 tracking - Q to exit', canvas)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        stopped_early = True
                        break
                if args.max_frames and frames >= args.max_frames:
                    stopped_early = True
                    break
                ok, frame = capture.read()
        elapsed = time.perf_counter() - started
        summary = dict(source=args.source, model=args.model, tracker=args.tracker, device=device,
                       frames=frames, source_fps=fps, width=width, height=height,
                       unique_track_ids=len(seen), elapsed_seconds=round(elapsed, 3),
                       processing_fps=round(frames / elapsed, 2), stopped_early=stopped_early,
                       classes=None if args.all_classes else args.classes, conf=args.conf,
                       imgsz=args.imgsz, ultralytics=ultralytics.__version__, torch=torch.__version__)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'Finished: {args.output.resolve()}')
        return summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        run(parse_args())
    except (RuntimeError, OSError, ValueError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
