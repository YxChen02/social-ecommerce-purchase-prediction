# Social E-commerce Purchase Prediction

🏆 **First Prize Winner** - The 6th Shanghai Intercollegiate Open Data Challenge 

## Overview

This project addresses the challenge of predicting user purchase behavior in social e-commerce platforms (such as Xiaohongshu, Douyin/TikTok). Unlike traditional e-commerce where users follow a "search-compare-purchase" path, social e-commerce operates through a "content discovery-interaction-trust-conversion" funnel. This repository contains comprehensive data analysis, user segmentation, association rule mining, and predictive modeling for understanding and optimizing purchase conversion in social commerce environments.

## Problem Background

Social e-commerce has transformed the consumer journey by integrating content creation, social interaction, and shopping. Key characteristics include:

- **Content-driven discovery**: Users are influenced by posts, short videos, and community interactions
- **Social signals**: Likes, comments, shares, and follows significantly impact purchase decisions
- **Dynamic heterogeneity**: Rapid trend changes, diverse user preferences, and varying sensitivity to discounts
- **Low conversion rates**: Most impressions don't lead to purchases, requiring intelligent targeting

This project aims to identify high-conversion user segments, understand behavioral mechanisms, and provide actionable insights for content distribution and operational strategies.

## Dataset

The dataset contains **100,000 user purchase behavior records** with **31 features** and **1 binary target variable**.

### Key Characteristics:
- **Primary demographic**: Young female users (average age: 27, 63.8% female)
- **Distribution**: Price and interaction data show right-skewed distributions typical of social e-commerce
- **Source**: Real-world scenarios from platforms like Xiaohongshu and Douyin

### Feature Categories:

1. **User Features** (10 features)
   - Demographics: age, gender, user_level
   - Historical behavior: purchase_freq, total_spend, register_days
   - Social metrics: follow_num, fans_num

2. **Content/Product Features** (7 features)
   - Product attributes: price, discount_rate, category
   - Content quality: title_length, title_emo_score, img_count, has_video

3. **Social/Interaction Features** (6 features)
   - Engagement metrics: like_num, comment_num, share_num, collect_num
   - Relationship strength: is_follow_author

4. **User-Item Interaction Features** (5 features)
   - Purchase funnel: add2cart, coupon_received, coupon_used
   - Engagement depth: pv_count, last_click_gap

5. **Derived Features** (4 features)
   - Composite metrics: interaction_rate, purchase_intent, freshness_score, social_influence

6. **Target Variable**
   - `label`: Purchase conversion (0=No Purchase, 1=Purchase)

## Project Structure

```
social-ecommerce-purchase-prediction/
├── EDA.ipynb                    # Exploratory Data Analysis
├── BGMM_Clustering.ipynb        # User Segmentation using Bayesian Gaussian Mixture Models
├── Association_Rules.py         # Association Rule Mining with Statistical Significance Testing
├── CatBoost.ipynb               # Purchase Prediction Model
├── social_ecommerce_data.csv    # Dataset (100k records)
└── README.md                    # This file
```

## Methodology

### 1. Exploratory Data Analysis (EDA.ipynb)

### 2. User Segmentation (BGMM_Clustering.ipynb)
- **Algorithm**: Bayesian Gaussian Mixture Model (BGMM) with automatic cluster selection
- **Features**: Multi-dimensional analysis including user, content, social, and behavioral features
- **Outcome**: Identified distinct user personas:
  - Discount-sensitive users
  - High-interaction users
  - Add-to-cart focused users
  - Impulse buyers
- **Analysis**: Compared conversion rates, characteristic patterns, and operational strategies for each segment

### 3. Association Rule Mining (Association_Rules.py)
- **Advanced statistical approach**: Chi-square independence testing (p < 0.05)
- **Metrics**: Support, Confidence, Lift, Leverage, Conviction
- **Rules explored**: 
  - Interaction behaviors (like/comment/share/collect)
  - Action sequences (add-to-cart/coupon usage/browsing)
  - Purchase outcomes
- **Application**: Identified high-conversion feature combinations for content distribution strategies

### 4. Predictive Modeling (CatBoost.ipynb)
- **Algorithm**: CatBoost gradient boosting classifier
- **Performance**: 
  - ROC-AUC Score: 0.7765
  - F1-Score: 0.6591
  - Accuracy: 0.72
- **Features**: Handled categorical variables natively, robust to overfitting with early stopping

##  Key Findings

### Conversion Insights
- **Strongest predictors**: Add-to-cart action, coupon usage, and repeated browsing (pv_count)
- **Social signals**: Users who follow authors show significantly higher conversion rates
- **Content format**: Video content and high emotional title scores drive better engagement
- **Price sensitivity**: Discount rate >15% significantly improves conversion for price-sensitive segments

### User Segments
- **Core segments identified**: 4-6 distinct behavioral clusters with varying conversion patterns
- **High-value users**: Characterized by frequent purchases, high interaction rates, and strong author relationships
- **Opportunity segments**: Users with high browsing but low conversion represent optimization opportunities

### Actionable Rules
- **High-lift combinations**: "Like + Add-to-cart → Purchase" and "Follow Author + Coupon Used → Purchase"
- **Content strategy**: Video content with strong discounts performs best for new users
- **Timing**: Shorter last_click_gap correlates with higher conversion probability

## Business Recommendations

1. **Personalized Content Distribution**
   - Match content format (video vs. image) to user segment preferences
   - Prioritize high-intent users for premium content placement

2. **Conversion Optimization**
   - Implement targeted coupon strategies based on user sensitivity
   - Strengthen author-follower relationships to build trust

3. **Operational Efficiency**
   - Reduce impressions to low-intent segments to improve platform efficiency
   - Focus on nurturing high-potential users through the purchase funnel

4. **Content Strategy**
   - Optimize title emotional scores and video content production
   - Leverage social proof through visible engagement metrics

## Results Summary

- **Model Performance**: CatBoost achieved 0.7765 ROC-AUC with strong generalization
- **User Segments**: Successfully identified 8 meaningful behavioral clusters
- **Association Rules**: Discovered 10+ statistically significant rules (p < 0.05) with lift > 1.5
- **Business Impact**: Insights directly applicable to content recommendation, user targeting, and conversion optimization
