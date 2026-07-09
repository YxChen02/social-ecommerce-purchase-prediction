import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ==========================================
# 0. 基础设置
# ==========================================
# 设置中文显示，防止图表中的中文变成方块
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("🚀 正在启动高阶关联规则分析任务 (包含卡方检验 & 复杂统计量)...")
    
    # 1. 读取数据
    try:
        df = pd.read_csv('social_ecommerce_data.csv')
        print(f"✅ 成功读取数据，共 {len(df)} 条记录。")
    except FileNotFoundError:
        print("❌ 错误：未找到文件 social_ecommerce_data.csv，请检查路径。")
        return

    # ==========================================
    # 2. 业务特征精细化二值化 (Feature Discretization)
    # ==========================================
    df_bin = pd.DataFrame()

    # --- A. 互动特征 ---
    df_bin['互动_有点赞'] = (df['like_num'] > 0).astype(int)
    df_bin['互动_有评论'] = (df['comment_num'] > 0).astype(int)
    df_bin['互动_有分享'] = (df['share_num'] > 0).astype(int)
    df_bin['互动_关注作者'] = df['is_follow_author']
    
    # --- B. 动作特征 ---
    df_bin['动作_已加购'] = df['add2cart']
    df_bin['动作_已领券'] = df['coupon_received']
    df_bin['动作_已用券'] = df['coupon_used']
    
    # --- C. 内容分发特征 (紧扣赛题) ---
    df_bin['内容_有视频'] = df['has_video']
    df_bin['内容_强折扣'] = (df['discount_rate'] > 0.15).astype(int) # 折扣率>15%
    df_bin['内容_高情感标题'] = (df['title_emo_score'] > 0.6).astype(int) # 情绪分>0.6

    # --- D. 目标变量 ---
    df_bin['结果_购买'] = df['label']

    # ==========================================
    # 3. 核心统计学引擎 (计算置信度、提升度、杠杆率、确信度、P值)
    # ==========================================
    def calculate_advanced_metrics(df, condition_col, target_col='结果_购买'):
        N = len(df)
        support_A = df[condition_col].mean()
        support_B = df[target_col].mean()
        support_AB = (df[condition_col] & df[target_col]).mean()
        
        # 如果前置条件发生概率为0，直接返回0和不显著的P值
        if support_A == 0: 
            return 0, 0, 0, 0, 0, 1.0 
            
        confidence = support_AB / support_A
        lift = confidence / support_B if support_B > 0 else 0
        
        # 杠杆率 (Leverage): 衡量实际共现与随机共现的绝对差值
        leverage = support_AB - (support_A * support_B)
        
        # 确信度 (Conviction): 衡量 B 对 A 的依赖程度
        if confidence == 1.0:
            conviction = float('inf')
        else:
            conviction = (1 - support_B) / (1 - confidence)
            
        # --- 核心革新：卡方检验 (Chi-square Test) ---
        # 构建 2x2 列联表，评估特征关联的统计显著性
        O_11 = max(0, support_AB * N)                     # A发生，B发生
        O_12 = max(0, (support_A - support_AB) * N)       # A发生，B未发生
        O_21 = max(0, (support_B - support_AB) * N)       # A未发生，B发生
        O_22 = max(0, (1 - support_A - support_B + support_AB) * N) # A未发生，B未发生
        
        contingency_table = [[O_11, O_12], [O_21, O_22]]
        
        try:
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        except:
            p_value = 1.0 # 如果计算失败，视为不显著
            
        return support_AB, confidence, lift, leverage, conviction, p_value

    # ==========================================
    # 4. 挖掘特征组合规则
    # ==========================================
    interaction_cols = ['互动_有点赞', '互动_有评论', '互动_有分享', '互动_关注作者']
    action_cols = ['动作_已加购', '动作_已领券', '动作_已用券']
    content_cols = ['内容_有视频', '内容_强折扣', '内容_高情感标题']
    
    results = []
    print("⏳ 正在进行组合空间遍历与卡方独立性检验...")

    # 遍历: (内容/互动) + (动作) -> 购买
    for col1 in interaction_cols + content_cols:
        for col2 in action_cols:
            rule_name = f"{{{col1}}} + {{{col2}}}"
            df_bin[rule_name] = df_bin[col1] & df_bin[col2]
            
            supp, conf, lift, lev, conv, p_val = calculate_advanced_metrics(df_bin, rule_name)
            
            # 严格过滤标准：支持度 > 1% 且 卡方检验显著 (P < 0.05)
            if supp > 0.01 and p_val < 0.05:
                results.append({
                    '挖掘规则': f"{rule_name} -> {{购买}}",
                    '支持度 (Support)': round(supp, 4),
                    '置信度 (Confidence)': round(conf, 4),
                    '提升度 (Lift)': round(lift, 4),
                    '杠杆率 (Leverage)': round(lev, 4),
                    '确信度 (Conviction)': round(conv, 4) if conv != float('inf') else '强依赖',
                    'P-value': "{:.2e}".format(p_val) # 科学计数法显示 P 值
                })

    # ==========================================
    # 5. 结果排序、输出与保存
    # ==========================================
    rules_df = pd.DataFrame(results).sort_values(by='提升度 (Lift)', ascending=False)
    print("\n=== 🏆 统计学显著的高阶关联规则 (Top 10) ===")
    print(rules_df.head(10))
    
    # 保存为 CSV
    csv_filename = "Ultimate_Association_Rules.csv"
    rules_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"📁 详细结果已保存至: {csv_filename}")

    # ==========================================
    # 6. 高阶可视化：学术级矩阵气泡图
    # ==========================================
    plt.figure(figsize=(12, 8))
    
    # 图表映射：X轴=支持度, Y轴=置信度, 气泡大小=提升度, 气泡颜色=杠杆率
    sizes = rules_df['提升度 (Lift)'] * 400
    scatter = sns.scatterplot(x="支持度 (Support)", y="置信度 (Confidence)", 
                              size=sizes, hue="杠杆率 (Leverage)", 
                              data=rules_df, palette="YlOrRd", 
                              sizes=(100, 1000), alpha=0.85, edgecolor='black')
    
    # 标注 Top 5 最核心的规则
    for i in range(min(5, len(rules_df))):
        row = rules_df.iloc[i]
        rule_text = row['挖掘规则'].split(' ->')[0]
        plt.text(row['支持度 (Support)'] + 0.002, row['置信度 (Confidence)'], 
                 rule_text, fontsize=10, fontweight='bold', color='#333333')
                 
    # 完善图表装饰
    plt.title('显著性关联规则空间分布图 (P<0.05)', fontsize=16, pad=20)
    plt.xlabel('支持度 (Support) - 规则触发频率', fontsize=12)
    plt.ylabel('置信度 (Confidence) - 购买转化概率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 调整图例
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="指标图例")
    
    plt.tight_layout()
    img_filename = 'Ultimate_Rules_Bubble.png'
    plt.savefig(img_filename, dpi=300)
    print(f"📊 高阶气泡图已保存至: {img_filename}")
    plt.show()

if __name__ == "__main__":
    main()