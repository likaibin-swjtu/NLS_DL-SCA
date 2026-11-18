# attack20_triplet_profiling_traces.py
import sys
import os
import time  # 🔴 添加 time 模块用于耗时统计
import subprocess  # 🔴 添加 subprocess 模块用于显存获取备用方案
from keras.models import load_model
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint

## for visualizing
import matplotlib
import matplotlib.pyplot as plt, numpy as np
from sklearn.preprocessing import StandardScaler
import Util.SCA_dataset as datasets
import Util.Attack as Attack
import Util.DL_model as DL_model
import Util.Template_attack as TA
import Util.Triplet_loss as losses
import Util.plot_profiling_traces as p_pt
from Util.basic_func import write2file
import tensorflow as tf
from datetime import datetime  # 添加时间记录

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(e)


# 🔴 Keras 回调：用于记录每个 Epoch 的耗时
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)


# 🔴 辅助函数：获取当前 GPU 显存使用量 (GB)
def get_gpu_memory():
    try:
        # 首选: 使用 TF >= 2.4 的内置功能获取峰值显存
        info = tf.config.experimental.get_memory_info('GPU:0')
        return info['peak'] / (1024 ** 3)
    except Exception:
        try:
            # 备选: 调用 nvidia-smi 命令行获取整体显存使用量
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                encoding='utf-8')
            return float(result.strip().split('\n')[0]) / 1024.0
        except Exception:
            return 0.0


# 🔴 改为测试不同的 profiling_traces 数量
profiling_traces_combinations = [
    # 10000,
    # 15000,  # 15K
    # 20000,  # 20K
    # 25000,  # 25K
    # 30000,  # 30K
    # 35000,  # 35K
    # 40000,  # 40K
    # 45000,  # 45K
    50000,  # 50K
]

# 🔴 将 leakage_models, embedding_size, epochs 也放入超参数字典中进行统一管理
best_hyperparams = {
    # 示例注释，按需打开
    # 'ASCAD': {'alpha_value': 0.1, 'margin': 0.4, 'leakage_models': ['HW'], 'embedding_size': 32, 'epochs': 2},  # ASCAD_HW_e2
    # 'ASCAD': {'alpha_value': 0.4, 'margin': 0.6, 'leakage_models': ['ID'], 'embedding_size': 32, 'epochs': 2},  # ASCAD_ID_e2
    # 'ASCAD_rand': {'alpha_value': 0.1, 'margin': 0.4, 'leakage_models': ['HW'], 'embedding_size': 32, 'epochs': 5},  # ASCAD_R_HW_e5
    'ASCAD_rand': {'alpha_value': 0.2, 'margin': 0.6, 'leakage_models': ['ID'], 'embedding_size': 32, 'epochs': 5},  # ASCAD_R_ID_e5
    # 'AESHD': {'alpha_value': 0.5, 'margin': 0.6, 'leakage_models': ['HW'], 'embedding_size': 16, 'epochs': 2},
}


def save_experiment_log(dataset, leakage_model, noise_type, noise_level, config_results, final_summary, log_dir):
    """
    🔴 保存实验日志到文件
    参数:
        dataset: 数据集名称
        leakage_model: 泄漏模型
        noise_type: 噪声类型
        noise_level: 噪声级别
        config_results: 每个配置的结果列表
        final_summary: 最终总结字典
        log_dir: 日志保存目录
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_filename = f"NLS_profiling_study_{dataset}_{leakage_model}_{noise_type}{noise_level}_log.txt"
    log_path = os.path.join(log_dir, log_filename)

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    with open(log_path, 'w', encoding='utf-8') as f:
        # 写入头部信息
        f.write("=" * 100 + "\n")
        f.write("NLS LOSS PROFILING TRACES IMPACT STUDY - EXPERIMENT LOG\n")
        f.write("=" * 100 + "\n")
        f.write(f"Experiment Time: {timestamp}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Leakage Model: {leakage_model}\n")
        f.write(f"Noise: {noise_type}={noise_level}\n")
        f.write(f"Total Configurations: {len(config_results)}\n")
        f.write(f"Repeats per Configuration: {config_results[0]['num_repeats'] if config_results else 'N/A'}\n")
        f.write("=" * 100 + "\n\n")

        # 写入每个配置的详细结果
        f.write("DETAILED RESULTS BY CONFIGURATION:\n")
        f.write("-" * 100 + "\n")

        for i, result in enumerate(config_results):
            f.write(f"\nConfiguration {i + 1}: {result['profiling_traces']} Profiling Traces\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Parameters:\n")
            f.write(f"  - Profiling Traces: {result['profiling_traces']}\n")
            f.write(f"  - Alpha Value: {result['alpha_value']}\n")
            f.write(f"  - Margin: {result['margin']}\n")
            f.write(f"  - Beta: {result['beta']}\n")
            f.write(f"  - Embedding Size: {result['embedding_size']}\n")
            f.write(f"  - Epochs: {result['epochs']}\n")
            f.write(f"  - Batch Size: {result['batch_size']}\n")
            f.write(f"\nPerformance Metrics:\n")  # 🔴 新增耗时与显存日志
            f.write(f"  - Avg Time/Epoch: {result['mean_time_per_epoch']:.2f} s\n")
            f.write(f"  - Avg VRAM Usage: {result['mean_vram_gb']:.2f} GB\n")
            f.write(f"\nResults:\n")
            f.write(f"  - Successful Runs: {result['successful_runs']}/{result['num_repeats']}\n")
            f.write(f"  - Success Rate: {result['success_rate']:.1f}%\n")
            f.write(f"  - Median TGE0: {result['median_tge0']}\n")
            f.write(f"  - Mean TGE0: {result['mean_tge0']:.2f}\n")
            f.write(f"  - TGE0 Std: {result['std_tge0']:.2f}\n")
            f.write(f"  - TGE0 Range: [{result['min_tge0']}, {result['max_tge0']}]\n")
            f.write(f"  - Median Final GE: {result['median_final_ge']:.2f}\n")
            f.write(f"  - All TGE0 Values: {result['tge0_all_values']}\n")
            f.write(f"  - Test Info: {result['test_info']}\n")
            f.write("-" * 60 + "\n")

        # 写入最终总结
        f.write(f"\nFINAL SUMMARY:\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best Configuration: {final_summary['best_config']['profiling_traces']} traces "
                f"(Median TGE0: {final_summary['best_config']['median_tge0']})\n")
        f.write(f"Worst Configuration: {final_summary['worst_config']['profiling_traces']} traces "
                f"(Median TGE0: {final_summary['worst_config']['median_tge0']})\n")
        f.write(f"Overall TGE0 Range: [{final_summary['overall_min_tge0']}, {final_summary['overall_max_tge0']}]\n")
        f.write(f"Average Success Rate: {final_summary['avg_success_rate']:.1f}%\n")

        f.write(f"\nSummary by Profiling Traces:\n")
        f.write("-" * 40 + "\n")
        for prof_traces, summary in final_summary['by_profiling_traces'].items():
            f.write(f"{prof_traces // 1000}K traces: Median TGE0 = {summary['median']:.0f}, "
                    f"Mean = {summary['mean']:.1f}, Success = {summary['success_rate']:.0f}%\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF EXPERIMENT LOG\n")
        f.write("=" * 100 + "\n")

    print(f"📝 Experiment log saved to: {log_path}")


if __name__ == "__main__":
    data_root = 'Data/'
    model_root = 'Model/triplet_profiling_study/'
    result_root = 'Result/triplet_profiling_study/'

    # 🔴 确保目录存在
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(result_root, exist_ok=True)

    # 🔴 数据集列表现在由字典的键自动决定
    datasetss = list(best_hyperparams.keys())
    classifier = 'NLS'
    noise_type = 'desync'
    noise_level = 0

    beta = 0.002
    batch_size = 512
    train_flag = True
    nb_attacks = 10
    num_repeats = 2  # Number of independent runs

    plt.rcParams.update({'font.size': 12})

    # 🔴 存储所有 profiling_traces 下的性能结果
    profiling_results = {}  # {prof_traces: [tge0_values]}
    all_results = []  # 存储详细结果用于可视化
    config_results = []  # 🔴 存储每个配置的结果用于日志记录

    for dataset in datasetss:
        print(f"\n🔥 Starting evaluation for dataset: {dataset}")

        # 🔴 动态获取该数据集的最优超参数
        if dataset in best_hyperparams:
            alpha_value = best_hyperparams[dataset]['alpha_value']
            margin = best_hyperparams[dataset]['margin']
            current_leakage_models = best_hyperparams[dataset]['leakage_models']
            embedding_size = best_hyperparams[dataset]['embedding_size']
            epochs = best_hyperparams[dataset]['epochs']
            print(
                f"Using best hyperparameters: alpha={alpha_value}, margin={margin}, leakage={current_leakage_models}, embed={embedding_size}, ep={epochs}")
        else:
            # 默认配置后备
            alpha_value = 0.2
            margin = 0.8
            current_leakage_models = ['HW']
            embedding_size = 16
            epochs = 5
            print(f"Using default hyperparameters: alpha={alpha_value}, margin={margin}")

        # 🔴 遍历当前数据集指定的泄漏模型
        for leakage_model in current_leakage_models:
            print(f"\n🎯 Testing {leakage_model} Leakage Model")
            print(f"📊 Testing {len(profiling_traces_combinations)} profiling traces configurations...")

            # 🔴 遍历不同的 profiling_traces 数量
            for config_idx, prof_traces in enumerate(profiling_traces_combinations):
                print(
                    f"\n📊 Configuration {config_idx + 1}/{len(profiling_traces_combinations)}: {prof_traces} profiling traces...")

                # 加载数据集
                if dataset == 'ASCAD':
                    correct_key = 224
                    nb_traces_attacks = 10000
                    (X_profiling, X_attack), (Y_profiling_ID, Y_attack_ID), (
                        plt_profiling, plt_attack) = datasets.load_ascad(
                        data_root + dataset + '/', profiling_traces=prof_traces, leakage_model='ID')
                elif dataset == 'ASCAD_rand':
                    correct_key = 34
                    nb_traces_attacks = 10000
                    (X_profiling, X_attack), (Y_profiling_ID, Y_attack_ID), (
                        plt_profiling, plt_attack) = datasets.load_ascad_rand(data_root + dataset + '/',
                                                                              profiling_traces=prof_traces,
                                                                              leakage_model='ID')
                elif dataset == 'AESHD':
                    correct_key = 200
                    nb_traces_attacks = 5000
                    (X_profiling, X_attack), (Y_profiling_ID, Y_attack_ID), (
                        plt_profiling, plt_attack) = datasets.load_aeshd(
                        data_root + dataset + '/', profiling_traces=prof_traces, leakage_model='ID')

                if noise_type == 'gnoise':
                    X_profiling = datasets.addGussianNoise(X_profiling, noise_level)
                    X_attack = datasets.addGussianNoise(X_attack, noise_level)
                if noise_type == 'desync':
                    X_profiling = datasets.addDesync(X_profiling, int(noise_level))
                    X_attack = datasets.addDesync(X_attack, int(noise_level))

                # Standardize and reshape data
                input_shape = (len(X_profiling[0]), 1)
                scaler = StandardScaler()
                X_profiling = scaler.fit_transform(X_profiling)
                X_attack = scaler.transform(X_attack)
                X_profiling = np.reshape(X_profiling, (len(X_profiling), X_profiling.shape[1], 1))
                X_attack = np.reshape(X_attack, (len(X_attack), X_attack.shape[1], 1))

                if leakage_model == 'HW':
                    num_classes = 9
                    Y_profiling = datasets.calculate_HW(Y_profiling_ID)
                    Y_attack = datasets.calculate_HW(Y_attack_ID)
                else:
                    num_classes = 256
                    Y_profiling = Y_profiling_ID
                    Y_attack = Y_attack_ID

                test_info = f'{dataset}_{leakage_model}_ep{epochs}_BS{batch_size}_prof{prof_traces}_alpha{alpha_value}_margin{margin}_beta{beta}'
                print(f'Test info: {test_info}')

                # 🔴 存储当前 profiling_traces 下的所有 TGE0 值
                tge0_values_for_current_traces = []

                # Store metrics for all runs
                all_final_ge = []
                all_ge_smaller_than_1 = []
                all_ge_smaller_than_5 = []
                all_ge_smaller_than_10 = []

                # 🔴 存储所有的耗时和显存记录
                all_times = []
                all_vrams = []

                for run in range(num_repeats):
                    print(f'Run {run + 1}/{num_repeats}...')

                    if train_flag:
                        # 使用原有的简单模型构建方式
                        model = DL_model.pick(classifier, input_shape, embedding_size)

                        # train session - 保持原有的训练流程 (注意这里依然是你修改的 lifted_struct_loss)
                        opt = Adam(learning_rate=5e-4)
                        loss = losses.lifted_struct_loss(alpha_value, margin, num_classes)
                        model.compile(loss=loss, optimizer=opt, metrics=['acc'])

                        filepath = model_root + 'NLS_model_{}_run{}.hdf5'.format(test_info, run + 1)
                        checkpoint = ModelCheckpoint(
                            filepath=filepath,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True)

                        # 🔴 加入记录时间的 Callback
                        time_callback = TimeHistory()
                        callbacks_list = [checkpoint, time_callback]

                        h = model.fit(
                            x=X_profiling,
                            y=Y_profiling,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_attack, Y_attack),
                            callbacks=callbacks_list,
                            verbose=2)

                        # 🔴 计算当前 run 的平均 Time/Epoch 和峰值显存 VRAM
                        avg_time_per_epoch = np.mean(time_callback.times) if time_callback.times else 0.0
                        vram_usage = get_gpu_memory()

                        print(
                            f'⏱️  [Profiling] Time/Epoch: {avg_time_per_epoch:.2f} s | VRAM Peak: {vram_usage:.2f} GB')
                        all_times.append(avg_time_per_epoch)
                        all_vrams.append(vram_usage)
                    else:
                        all_times.append(0.0)
                        all_vrams.append(0.0)

                    # 加载模型 - 保持原有的加载方式
                    model_path = model_root + 'NLS_model_{}_run{}.hdf5'.format(test_info, run + 1)
                    model = load_model(model_path, compile=False)

                    print('======Template attack======')
                    try:
                        # 使用嵌入向量进行攻击 - 保持原有的攻击流程
                        x_profiling_embeddings = model.predict(X_profiling)
                        x_embeddings = model.predict(X_attack)

                        mean_v, cov_v, classes = TA.template_training(x_profiling_embeddings, Y_profiling, pool=False)
                        TA_prediction = TA.template_attacking_proba(mean_v, cov_v, x_embeddings, classes)
                        avg_rank, all_rank = Attack.perform_attacks(nb_traces_attacks, TA_prediction, correct_key,
                                                                    plt_attack,
                                                                    log=True, dataset=dataset, nb_attacks=nb_attacks)

                        # Store the final Guessing Entropy (GE) value for each run
                        all_final_ge.append(avg_rank[-1])

                        # Store GE smaller than specific thresholds
                        ge_smaller_1 = np.argmax(avg_rank < 1) if np.any(avg_rank < 1) else len(avg_rank)
                        ge_smaller_5 = np.argmax(avg_rank < 5) if np.any(avg_rank < 5) else len(avg_rank)
                        ge_smaller_10 = np.argmax(avg_rank < 10) if np.any(avg_rank < 10) else len(avg_rank)

                        all_ge_smaller_than_1.append(ge_smaller_1)
                        all_ge_smaller_than_5.append(ge_smaller_5)
                        all_ge_smaller_than_10.append(ge_smaller_10)

                        # 🔴 收集 TGE0 值
                        tge0_values_for_current_traces.append(ge_smaller_1)

                        print(f'GE: {avg_rank[-1]}')
                        print(f'TGE0: {ge_smaller_1}')

                    except np.linalg.LinAlgError as e:
                        print(e)
                        print('Template attack error, singular matrix maybe')

                # 🔴 存储当前 profiling_traces 的所有 TGE0 值
                profiling_results[prof_traces] = tge0_values_for_current_traces

                # Calculate median metrics - 保持原有的统计方式
                median_final_ge = np.median(all_final_ge)
                median_ge_smaller_1 = int(np.median(all_ge_smaller_than_1))
                mean_ge_smaller_1 = np.mean(all_ge_smaller_than_1)
                std_ge_smaller_1 = np.std(all_ge_smaller_than_1) if len(all_ge_smaller_than_1) > 1 else 0
                min_tge0 = int(np.min(all_ge_smaller_than_1)) if all_ge_smaller_than_1 else 0
                max_tge0 = int(np.max(all_ge_smaller_than_1)) if all_ge_smaller_than_1 else 0
                median_ge_smaller_5 = int(np.median(all_ge_smaller_than_5))
                median_ge_smaller_10 = int(np.median(all_ge_smaller_than_10))

                # 🔴 计算耗时和显存的平均值
                mean_time_per_epoch = np.mean(all_times) if all_times else 0.0
                mean_vram_gb = np.mean(all_vrams) if all_vrams else 0.0

                # Print results for current profiling_traces
                print(f'\n===== Results for {prof_traces} Profiling Traces =====')
                print(f'Avg Time/Epoch: {mean_time_per_epoch:.2f} s')  # 🔴 输出耗时
                print(f'Avg VRAM Usage: {mean_vram_gb:.2f} GB')  # 🔴 输出显存
                print(f'Median Final GE: {median_final_ge}')
                print(f'Median TGE0: {median_ge_smaller_1}')
                print(f'Mean TGE0: {mean_ge_smaller_1:.2f}')
                print(f'TGE0 Std: {std_ge_smaller_1:.2f}')
                print(f'TGE0 Range: [{min_tge0}, {max_tge0}]')
                print(
                    f'Success Rate: {len(tge0_values_for_current_traces)}/{num_repeats} ({len(tge0_values_for_current_traces) / num_repeats * 100:.0f}%)')

                # 🔴 存储配置结果用于日志记录
                config_result = {
                    'config_idx': config_idx + 1,
                    'profiling_traces': prof_traces,
                    'alpha_value': alpha_value,
                    'margin': margin,
                    'beta': beta,
                    'embedding_size': embedding_size,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'num_repeats': num_repeats,
                    'successful_runs': len(tge0_values_for_current_traces),
                    'success_rate': len(tge0_values_for_current_traces) / num_repeats * 100,
                    'median_final_ge': median_final_ge,
                    'median_tge0': median_ge_smaller_1,
                    'mean_tge0': mean_ge_smaller_1,
                    'std_tge0': std_ge_smaller_1,
                    'min_tge0': min_tge0,
                    'max_tge0': max_tge0,
                    'tge0_all_values': tge0_values_for_current_traces,
                    'mean_time_per_epoch': mean_time_per_epoch,  # 🔴 记录到字典
                    'mean_vram_gb': mean_vram_gb,  # 🔴 记录到字典
                    'test_info': test_info
                }
                config_results.append(config_result)

                # 🔴 为可视化准备详细结果
                result_entry = {
                    'config_id': config_idx + 1,
                    'leakage_model': leakage_model,
                    'profiling_traces': prof_traces,
                    'alpha_value': alpha_value,
                    'margin': margin,
                    'beta': beta,
                    'embedding_size': embedding_size,
                    'epochs': epochs,
                    'tge0_median': median_ge_smaller_1,
                    'tge0_avg': mean_ge_smaller_1,
                    'tge0_std': std_ge_smaller_1,
                    'tge0_min': min_tge0,
                    'tge0_max': max_tge0,
                    'tge0_all_values': tge0_values_for_current_traces,
                    'successful_runs': len(tge0_values_for_current_traces),
                }
                all_results.append(result_entry)

        # 🔴 准备最终总结数据用于日志记录
        final_summary = {
            'best_config': min(config_results, key=lambda x: x['median_tge0']),
            'worst_config': max(config_results, key=lambda x: x['median_tge0']),
            'overall_min_tge0': min(config_results, key=lambda x: x['min_tge0'])['min_tge0'],
            'overall_max_tge0': max(config_results, key=lambda x: x['max_tge0'])['max_tge0'],
            'avg_success_rate': np.mean([r['success_rate'] for r in config_results]),
            'by_profiling_traces': {}
        }

        # 按profiling_traces总结
        prof_traces_sorted = sorted(profiling_results.keys())
        for prof_traces in prof_traces_sorted:
            tge0_values = profiling_results[prof_traces]
            if tge0_values:
                final_summary['by_profiling_traces'][prof_traces] = {
                    'median': np.median(tge0_values),
                    'mean': np.mean(tge0_values),
                    'success_rate': len(tge0_values) / num_repeats * 100
                }

        # 🔴 保存实验日志
        save_experiment_log(dataset, leakage_model, noise_type, noise_level,
                            config_results, final_summary, result_root)

        # 🔴 生成性能可视化图表 - 使用原有的数据结构
        print(f"\n🎨 Generating performance visualizations...")
        try:
            save_dir, analysis_results = p_pt.analyze_profiling_traces_convergence(
                all_results,
                dataset=dataset,
                leakage_model=leakage_model,
                noise_type=noise_type,
                noise_level=noise_level
            )
            print(f"✅ Analysis complete! All visualizations saved to: {save_dir}")
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")

        # 🔴 打印最终总结
        print(f"\n===== Final Summary for {dataset} =====")
        prof_traces_sorted = sorted(profiling_results.keys())
        for prof_traces in prof_traces_sorted:
            tge0_values = profiling_results[prof_traces]
            if tge0_values:  # 确保列表不为空
                median_tge0 = np.median(tge0_values)
                mean_tge0 = np.mean(tge0_values)
                success_rate = len(tge0_values) / num_repeats * 100
                print(
                    f'{prof_traces // 1000}K traces: Median TGE0 = {median_tge0:.0f}, Mean = {mean_tge0:.1f}, Success = {success_rate:.0f}%')

    print(f"\n✅ All visualizations saved to: {result_root}")
    print('\n' + '=' * 80)
    print('NLS PROFILING TRACES EXPERIMENT COMPLETED')
    print('=' * 80)