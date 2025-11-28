import os                                              # 操作系统相关模块（路径、文件等）
import torch                                           # PyTorch 深度学习框架
from models import GPT4ts,Model4F,Bert_v1,GPT4ts_v1,GPT4ts_v2,GPT4ts_v3,GPT4ts_v4,qwen4ts,qwen4ts_v1,qwen4ts_v2,qwen4ts_linear,qwen4ts_v3  # 导入各种模型类
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,6"     # 指定可见 GPU（注释掉）
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs   # 导入 HuggingFace Accelerate 的加速组件


class Exp_Basic(object):                               # 定义一个基础实验类（所有 Exp_xxx 类的父类）
    def __init__(self, args):                          # 初始化函数，接收 args 对象
        self.args = args                               # 保存配置参数

        # ========= 分布式与 DeepSpeed 插件配置 =========
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)   # 允许未使用的参数，避免 DDP 报错
        args.exp_name = "bert_vq_96to96_lr1e-4"         # 默认实验名称
        args.log_with = ["tensorboard"]                 # 使用 tensorboard 作为日志后端

        assert args.deepspeed_config_path.endswith('.json') and os.path.exists(args.deepspeed_config_path), \
            f"Invalid DeepSpeed config path: {args.deepspeed_config_path}"       # 检查 DeepSpeed 配置文件是否有效
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config_path)  # 创建 DeepSpeed 插件实例

        # ========= 日志路径定义 =========
        exp_log_dir = os.path.join(args.checkpoints, 'logs', args.exp_name)      # 日志路径 = checkpoints/logs/exp_name

        # ========= 多日志平台支持（tensorboard / wandb 等） =========
        loggers = args.log_with if isinstance(args.log_with, list) else [args.log_with]  # 统一变成列表

        # ========= Accelerator 初始化 =========
        self.accelerator = Accelerator(               # 初始化 Accelerate
            kwargs_handlers=[ddp_kwargs],             # 传入 DDP 参数
            deepspeed_plugin=deepspeed_plugin,        # 注入 DeepSpeed 插件
            log_with=loggers,                         # 记录日志到 tensorboard 或 wandb
            project_dir=exp_log_dir                   # 日志存储目录
        )

        # ========= 当前进程 device =========
        self.device = self.accelerator.device         # 当前 rank（进程）使用的 device（cpu/gpu）

        # ========= 调试信息 =========
        if self.accelerator.is_local_main_process:    # 仅主进程打印日志
            print("=" * 50)
            print(f"[Accelerator] Initialized on device: {self.device}")         # 当前设备
            print(f"[Accelerator] Logging to: {exp_log_dir}")                    # 日志路径
            print(f"[Accelerator] Log backends: {loggers}")                      # 日志后端
            print("=" * 50)

        self.model_dict = {                           # 模型名称映射字典
            'Bert4ts': Bert_v1,                       # BERT 模型
            'GPT4ts': GPT4ts,                         # GPT4ts 模型
            'Model4F': Model4F,                       # 自定义模型 Model4F
            'GPT4ts1': GPT4ts_v1,                     # GPT4ts v1
            'GPT4ts2': GPT4ts_v2,                     # GPT4ts v2
            'GPT4ts3': GPT4ts_v3,                     # GPT4ts v3
            'GPT4ts4': GPT4ts_v4,                     # GPT4ts v4
            'qwen4ts': qwen4ts,                       # Qwen 模型
            'qwen4ts1': qwen4ts_v1,                   # Qwen v1
            'qwen4ts2': qwen4ts_v2,                   # Qwen v2
            'linear': qwen4ts_linear,                 # 线性版本模型
            'qwen4ts3': qwen4ts_v3                    # Qwen v3
        }
        self.model, self.vq_model, self.classification_weight = self._build_model()   # 构建模型（由子类实现）
        # self.model = self._build_model()            # 如果只返回一个模型，可用此行（当前注释掉）

    def _build_model(self):                           # 构建模型，由子类重写
        raise NotImplementedError                     # 未实现则报错
        return None                                   # 占位返回（不可达）

    def _acquire_device(self):                        # 旧版获取设备的函数（已不使用）
        if self.args.use_gpu:                         # 使用 GPU？
            # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,6,7"    # 设置可见 GPU（注释）
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))  # 按参数选择 GPU
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')              # 使用 CPU
            print('Use CPU')
        return device                                  # 返回最终设备

    def _get_data(self):                               # 数据加载接口，子类实现
        pass

    def vali(self):                                    # 验证接口，子类实现
        pass

    def train(self):                                   # 训练接口，子类实现
        pass

    def test(self):                                    # 测试接口，子类实现
        pass
