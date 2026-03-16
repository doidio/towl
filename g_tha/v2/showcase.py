# uv run streamlit run v2/showcase.py --server.port 8500

import streamlit as st
import sys
from pathlib import Path
import time

# 将当前目录加入 sys.path 以确保能正确导入 v2.infer
sys.path.append(str(Path(__file__).parent))
from infer import main as infer_main

st.set_page_config('G-THA', initial_sidebar_state='expanded', layout='wide')
st.title('G-THA Showcase')

st.caption('生成式 AI 端到端预测准确填充匹配患者术前 CT 的 THA 股骨柄假体')

# --- 资源路径检查 ---
test_dir = Path(__file__).parent / "test"
test_dir.mkdir(parents=True, exist_ok=True)
test_files = sorted(list(test_dir.glob("*.nii.gz")))
file_names = [f.name for f in test_files]

# --- 侧边栏参数 ---
if not file_names:
    st.sidebar.error(f"请在以下目录放入测试文件:\n`{test_dir.resolve()}`")
    st.info("等待测试图像入库...")
    st.stop()

with st.sidebar:
    st.markdown('#### 生成条件')
    selected_file_name = st.selectbox("选择术前图像", file_names)

    st.markdown('#### 生成控制')
    cfg = st.slider('CFG 权重 (Guidance Scale)', 0.0, 10.0, 3.0, 0.5)
    ts = st.slider('采样步数 (Steps)', 50, 1000, 50, 50)

    use_seed = st.checkbox('固定随机种子', value=True)
    # 启用固定随机种子才可输入种子数值
    seed = st.number_input('随机种子', value=42, min_value=0, max_value=999999, disabled=not use_seed)

    st.markdown('#### 性能选项')
    amp = st.checkbox('启用混合精度 (AMP)', value=True)
    # 默认不启用分块推理
    tiled = st.checkbox('启用分块推理 (Tiled)', value=False)
    # 默认不启用摘要
    enable_summary = st.checkbox('生成过程摘要图', value=False)

    st.divider()
    submit_button = st.button('🚀 开始生成', use_container_width=True)

# --- 主界面 ---
result_placeholder = st.container()


# 将 ui_printf 修改为支持动态传入 placeholder
def ui_printf(text, placeholder):
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    st.session_state.logs.append(text)
    # 保持一定的日志长度，避免 UI 过长
    if len(st.session_state.logs) > 20:
        st.session_state.logs.pop(0)
    placeholder.code('\n'.join(st.session_state.logs), language='terminal')


if submit_button:
    st.session_state.logs = []

    # 固定的默认参数
    params = {
        'cond': str(test_dir / selected_file_name),
        'save': 'v2/save',
        'vae_pre': None,
        'vae_metal': None,
        'ldm': None,
        'cpu': False,
        'amp': amp,
        'sw': 4,
        'tiled': tiled,
        'seed': seed if use_seed else None,
        'cfg': cfg,
        'ts': ts,
        'summary': enable_summary,
    }

    start_time = time.time()
    # 将 Spinner 放在状态提示下方
    with st.spinner('AI 正在采样计算中...'):
        # log 放在 spinner 下方：在 spinner 容器内部创建一个 placeholder
        log_placeholder = st.empty()
        params['printf'] = lambda t: ui_printf(t, log_placeholder)

        infer_main(**params)

    end_time = time.time()
    st.success(f'✅ 生成完成！总耗时: {end_time - start_time:.2f}s')

    # 结果展示逻辑
    with result_placeholder:
        st.divider()
        cond_p = Path(params['cond'])
        save_name = '_'.join([
            cond_p.with_suffix('').with_suffix('').name,
            'seed', str(params['seed']) if use_seed else 'random',
            'cfg', str(cfg),
            'ts', str(ts),
            'tiled' if tiled else 'no-tiled',
            'summary' if params['summary'] else 'no-summary',
        ])
        actual_save_dir = Path(params['save']) / save_name

        if enable_summary:
            summary_img_path = actual_save_dir / cond_p.name.replace('.nii.gz', '_summary.png')
            if summary_img_path.exists():
                st.image(str(summary_img_path), caption=f'生成过程摘要 - {selected_file_name}')

        st.markdown(f"**结果目录:** `{actual_save_dir.resolve()}`")
        files = sorted(list(actual_save_dir.glob('*')))
        for f in files:
            if f.is_file() and f.suffix in ['.gz', '.stl', '.png']:
                st.write(f"📦 {f.name}")
else:
    st.info(f"💡 当前测试库中共计 {len(file_names)} 个文件。点击左侧按钮开始生成。")
