import argparse
from .config import load


def main():
    parser = argparse.ArgumentParser(description='Axis P5544 + YOLO: static stereo and calibrated independent PTZ tracking/ranging')
    parser.add_argument('--config',default='config.yaml')
    sub = parser.add_subparsers(dest='command',required=True)
    doctor = sub.add_parser('doctor',help='Port, JPEG, PTZ and RTSP decoder diagnostics')
    doctor.add_argument('--channels',type=int,nargs='+',default=[1])
    sub.add_parser('preview',help='Live streams without YOLO or calibration')
    sub.add_parser('capture',help='Capture stationary chessboard pairs')
    sub.add_parser('calibrate',help='Fit and validate intrinsics + stereo extrinsics')
    evaluation = sub.add_parser('evaluate',help='Evaluate rectified pixel correspondences against measured ranges')
    evaluation.add_argument('csv')
    sub.add_parser('manual',help='Measure a manually selected corresponding physical point')
    gpu = sub.add_parser('gpu', help='Verify CUDA availability and execute a real GPU calculation')
    gpu.add_argument('--device', default='0', help='Visible CUDA index, default 0')
    track = sub.add_parser('track', help='Select one person; independent PTZ with optional calibrated dynamic ranges')
    track.add_argument('--enable-ptz', action='store_true', help='Send bounded camera movement commands; default is dry run')
    track.add_argument('--max-frames', type=int)
    track.add_argument('--device', help='Override model.device: cpu or CUDA index such as 0')
    track.add_argument('--ranges',action='store_true',help='Load real PTZ calibration and measured timing profile')
    sub.add_parser('ptz-capture',help='Capture varied camera poses viewing ONE stationary chessboard')
    sub.add_parser('ptz-calibrate',help='Fit PTZ axes with held-out validation')
    timing = sub.add_parser('timing-calibrate',help='Fit empirical delay intervals from independent exposure measurements')
    timing.add_argument('csv')
    beacon = sub.add_parser('timing-beacon',help='Collect optical delay samples; both cameras must see this host display')
    beacon.add_argument('--display-lag-bound-ms',required=True,type=float,help='Independently measured upper bound including GUI/display pipeline')
    beacon.add_argument('--exposure-bound-ms',required=True,type=float,help='Known camera maximum exposure duration during this session')
    beacon.add_argument('--tick-ms',type=float,default=100.)
    beacon.add_argument('--samples',type=int,default=30)
    dynamic_eval = sub.add_parser('dynamic-evaluate',help='Compare tracking frame ranges with field ground truth')
    dynamic_eval.add_argument('log')
    dynamic_eval.add_argument('csv')
    run = sub.add_parser('run',help='YOLO detection and optional stereo measurement')
    run.add_argument('--detect-only',action='store_true')
    run.add_argument('--headless',action='store_true')
    run.add_argument('--max-frames',type=int)
    run.add_argument('--device', help='Override model.device: cpu or CUDA index such as 0')
    args = parser.parse_args()
    try:
        cfg = load(args.config)
        if getattr(args,'device',None) is not None:
            cfg['model']['device'] = args.device
        if args.command=='doctor':
            from .diagnostics import doctor
            doctor(cfg,args.channels)
        elif args.command=='preview':
            from .diagnostics import preview
            preview(cfg)
        elif args.command=='capture':
            from .calibration import capture
            capture(cfg)
        elif args.command=='calibrate':
            from .calibration import solve
            solve(cfg)
        elif args.command=='evaluate':
            from .evaluate import evaluate
            if not evaluate(cfg,args.csv)['all_samples_within_target']:
                return 2
        elif args.command=='manual':
            from .manual import measure
            measure(cfg)
        elif args.command=='run':
            from .runtime import run
            run(cfg,args.detect_only,args.headless,args.max_frames)
        elif args.command=='track':
            from .tracking import run
            run(cfg,args.enable_ptz,args.max_frames,args.ranges)
        elif args.command=='ptz-capture':
            from .dynamic_calibration import capture
            capture(cfg)
        elif args.command=='ptz-calibrate':
            from .dynamic_calibration import solve
            solve(cfg)
        elif args.command=='timing-calibrate':
            from .timing import calibrate
            calibrate(cfg,args.csv)
        elif args.command=='timing-beacon':
            from .timing_beacon import collect
            collect(cfg,args.display_lag_bound_ms,args.exposure_bound_ms,args.tick_ms,args.samples)
        elif args.command=='dynamic-evaluate':
            from .ranging import evaluate
            if not evaluate(cfg,args.log,args.csv)['all_samples_within_20cm']:
                return 2
        elif args.command=='gpu':
            from .compute import check_gpu
            check_gpu(cfg)
    except KeyboardInterrupt:
        print('Stopped by user')
    except Exception as exc:
        # ValueErrors raised by this package are actionable and do not contain URLs.
        # External/network/model errors may contain private URLs: output only class.
        if isinstance(exc,ImportError):
            print('ERROR: Missing/incompatible dependency:',exc.name or type(exc).__name__,
                  '| install requirements.txt in this Python environment')
        elif isinstance(exc,(ValueError,TimeoutError,FileNotFoundError)):
            print('ERROR:',str(exc))
        else:
            # Importing axis.py requires OpenCV; config/dependency errors must
            # still be printable when OpenCV itself is not installed.
            from .errors import redacted_error
            print('ERROR:',redacted_error(exc))
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
