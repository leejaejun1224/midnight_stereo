# utils.py
import os
from datetime import datetime
from typing import Dict, Any

def save_args_txt_dynamic(args, save_dir: str, filename: str = "args.txt") -> str:
    """
    - args의 모든 키를 동적으로 저장합니다(섹션 없음).
    - 헤더/구분선/정렬은 질문에서 주신 포맷을 그대로 따릅니다.
    - 기존 파일이 있으면 타임스탬프가 붙은 새 파일 이름으로 저장합니다.
    - bool은 True/False, None은 'None', float은 지수 표기 없이 깔끔히 출력합니다.
    """
    from datetime import datetime, timezone, timedelta

    os.makedirs(save_dir, exist_ok=True)

    # 타임스탬프 (KST)
    ts = datetime.now(tz=timezone.utc).astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%dT%H:%M:%S")

    # 파일 경로 (중복 시 타임스탬프 파일명)
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        ts_name = datetime.now(tz=timezone.utc).astimezone(timezone(timedelta(hours=9))).strftime("%y%m%d_%H%M%S")
        base, ext = os.path.splitext(filename)
        path = os.path.join(save_dir, f"{base}_{ts_name}{ext}")

    # 값 포맷터
    def _fmt(v):
        if isinstance(v, bool):
            return "True" if v else "False"
        if v is None:
            return "None"
        if isinstance(v, float):
            s = f"{v:.12f}".rstrip("0").rstrip(".")
            if s == "-0": s = "0"
            return s
        return str(v)

    kv = vars(args)
    keys = sorted(kv.keys())
    width = max(8, max(len(k) for k in keys)) if keys else 8

    sep = "=" * 80
    lines = [sep, f"Stereo Training Arguments — {ts}", sep, ""]
    for k in keys:
        lines.append(f"{k.ljust(width)} : {_fmt(kv[k])}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _stringify(v: Any) -> str:
    """값을 보기 좋게 문자열로 변환"""
    if isinstance(v, float):
        # 소수는 과도한 자리수를 줄여서 가독성↑
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_stringify(x) for x in v) + "]"
    return str(v)


def _format_block(title: str, items: Dict[str, Any]) -> str:
    """섹션(블록) 단위로 정렬된 문자열 생성"""
    if not items:
        return ""
    keys = list(items.keys())
    maxk = max(len(k) for k in keys) if keys else 0
    lines = [f"[{title}]"]
    for k in sorted(keys):
        lines.append(f"{k:<{maxk}} : {_stringify(items[k])}")
    lines.append("")  # 섹션 간 빈 줄
    return "\n".join(lines)


def _group_args(args_dict: Dict[str, Any]) -> str:
    """
    주요 카테고리로 묶어서 출력.
    args_dict에 없는 키는 자동으로 [Other] 섹션에 정리됩니다.
    """
    # 섹션 정의(필요 시 자유롭게 수정 가능)
    sections = [
        ("Data", ["left_dir", "right_dir", "mask_dir", "height", "width"]),
        ("Model", ["max_disp_px", "patch_size", "agg_ch", "agg_depth", "softarg_t", "norm"]),
        ("Optimization", ["epochs", "batch_size", "lr", "optim", "weight_decay", "workers", "amp"]),
        ("ROI", ["roi_method", "roi_thr"]),
        ("Directional Neighbor", ["sim_thr", "sim_gamma", "sim_sample_k",
                                  "use_dynamic_thr", "dynamic_q", "lambda_v", "lambda_h", "huber_delta_h"]),
        ("Sharpen Consistency", ["w_hsharp", "tau_sharp"]),
        ("Prob/Entropy", ["w_probcons", "w_entropy"]),
        ("Anchor/Reproj", ["w_anchor", "anchor_tau", "anchor_margin", "anchor_topk", "w_reproj"]),
        ("Photo/Smooth", ["w_photo", "w_smooth", "photo_l1_w", "photo_ssim_w"]),
        ("Enhance", ["no_enhance", "enhance_gamma", "enhance_clahe_clip", "enhance_clahe_tile"]),
        ("Resume/Checkpoint", ["resume", "resume_non_strict", "resume_reset_optim", "resume_reset_scaler"]),
        ("Logging", ["log_every", "save_dir"]),
        # Seeded Prior 1/8 (질문 코드에 포함된 옵션들)
        ("Seeded Prior (1/8)", [
            "w_seed", "seed_low_idx_thr", "seed_high_idx_thr", "seed_conf_thr",
            "seed_road_ymin", "seed_bin_w", "seed_min_count", "seed_tau", "seed_huber_delta"
        ]),
    ]

    used_keys = set()
    blocks = []

    # 정의된 섹션대로 묶기
    for title, keys in sections:
        subset = {k: args_dict[k] for k in keys if k in args_dict}
        used_keys.update(subset.keys())
        block = _format_block(title, subset)
        if block:
            blocks.append(block)

    # 그 외 키들(추가 파라미터가 있을 수 있음)
    others = {k: v for k, v in args_dict.items() if k not in used_keys}
    if others:
        blocks.append(_format_block("Other", others))

    return "\n".join(blocks)


def save_args_as_text(args, save_dir: str, filename: str = "trainlog.txt",
                      append_if_exists: bool = True,
                      title: str = "Stereo Training Arguments") -> str:
    """
    argparse.Namespace에 들어있는 파라미터를 보기 좋게 정리해 txt로 저장.
    - save_dir가 없으면 생성
    - 파일이 이미 있으면 기본적으로 append(추가 기록)
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)

    mode = "a" if (append_if_exists and os.path.exists(out_path)) else "w"
    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)

    header = []
    header.append("=" * 80)
    header.append(f"{title} — {datetime.now().isoformat(timespec='seconds')}")
    header.append("=" * 80)
    header_text = "\n".join(header)

    body_text = _group_args(args_dict)

    with open(out_path, mode, encoding="utf-8") as f:
        if mode == "a":
            f.write("\n")  # 기존 로그와 구분
        f.write(header_text + "\n\n" + body_text + "\n")

    return out_path
