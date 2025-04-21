from scipy.stats import ttest_ind, f_oneway

def compare_performance(results_dict):
    for model_name, scores in results_dict.items():
        print(f"{model_name}: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")

    # Example: t-test between PrototypicalNet and MAML
    t_stat, p_val = ttest_ind(results_dict['ProtoNet'], results_dict['MAML'])
    print(f"T-test between ProtoNet and MAML: p={p_val:.4f}")
