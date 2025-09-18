# Model Metadata Analysis Summary

## Overview
Analysis of 4,683 GPU-trained models from the quantitative trading system.

## 1. Accuracy Distribution

| Accuracy Range | Count | Percentage |
|---------------|-------|------------|
| >70% | 98 | 2.1% |
| 60-70% | 1,151 | 24.6% |
| 55-60% | 1,348 | 28.8% |
| 50-55% | 1,013 | 21.6% |
| <50% | 1,073 | 22.9% |
| **Total** | **4,683** | **100%** |

## 2. Top 100 Sweet Spot Models (60-70% Accuracy)

### Top 10 Examples:
1. **CINT**: 70.0% accuracy, +0.311 correlation
2. **LLYVA**: 69.8% accuracy, -0.000 correlation
3. **APWC**: 69.8% accuracy, +0.030 correlation
4. **AQMS**: 69.8% accuracy, -0.059 correlation
5. **ASO**: 69.8% accuracy, +0.212 correlation
6. **CRIS**: 69.8% accuracy, +0.100 correlation
7. **DFLI**: 69.8% accuracy, +0.016 correlation
8. **HSDT**: 69.8% accuracy, +0.141 correlation
9. **KXIN**: 69.8% accuracy, +0.029 correlation
10. **RDGT**: 69.8% accuracy, -0.142 correlation

## 3. Models with Good Accuracy & Low Correlation

**Found 794 models** with >=60% accuracy and |correlation| <= 0.2

### Top 10 Truly Predictive Models:
1. **DGLY**: 83.6% accuracy, -0.141 correlation
2. **ATMV**: 79.3% accuracy, +0.000 correlation
3. **APVO**: 79.0% accuracy, +0.082 correlation
4. **CDT**: 78.4% accuracy, -0.057 correlation
5. **KLTO**: 77.0% accuracy, -0.139 correlation
6. **ELWS**: 76.9% accuracy, -0.111 correlation
7. **LPTX**: 76.7% accuracy, +0.190 correlation
8. **WHLR**: 76.6% accuracy, +0.098 correlation
9. **GNLN**: 76.1% accuracy, +0.188 correlation
10. **IVF**: 76.1% accuracy, +0.002 correlation

## 4. Tiered Classification

### Tier 1: 212 models (65-70% accuracy with |correlation| <= 0.2)
**Premium models with high accuracy and low correlation**
- Accuracy range: 65.0% - 69.8%
- Average accuracy: 67.0%
- **Top examples**: LLYVA, APWC, AQMS, CRIS, DFLI

### Tier 2: 821 models (60-65% accuracy)
**Solid performers in the sweet spot**
- Accuracy range: 60.0% - 65.0%
- Average accuracy: 62.3%
- **Top examples**: GYRO, BBUC, TURB, ASNS, ACLS

### Tier 3: 1,348 models (55-60% accuracy)
**Moderate performers above random**
- Accuracy range: 55.0% - 59.9%
- Average accuracy: 57.5%
- **Top examples**: IMPP, MLYS, VBIX, ABSI, AIHS

## 5. Summary Statistics

- **Total models analyzed**: 4,683
- **Average accuracy**: 55.4%
- **Median accuracy**: 56.0%
- **Standard deviation**: 7.8%
- **Average correlation**: 0.019
- **Average |correlation|**: 0.144

## 6. Key Insights

- **98 models (2.1%)** achieve >70% accuracy - true outperformers
- **1,151 models (24.6%)** in the sweet spot (60-70%) - avoid trend-following
- **794 models** have both good accuracy (>=60%) and low correlation - truly predictive
- **Tier 1 contains 212** truly predictive models - the cream of the crop
- **1,073 models (22.9%)** perform worse than random - candidates for removal

## Recommendations

1. **Focus on Tier 1 models** (212 models) for core trading strategies
2. **Use Tier 2 models** (821 models) for diversification and portfolio filling
3. **Consider Tier 3 models** (1,348 models) for specific market conditions
4. **Remove underperforming models** (<50% accuracy) to reduce noise
5. **Prioritize low correlation models** in the 60%+ accuracy range for truly independent signals

## File Location
Analysis script: `C:\quant_system_v2\analyze_model_metadata.py`
Source data: `C:\quant_system_v2\quant_system_full\gpu_models\model_metadata.json`