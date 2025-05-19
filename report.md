# Tarot Reader Model Evaluation

## Model Overview
- **Algorithm**: Logistic Regression + TF-IDF
- **Categories**: Love, Career, Health, Finance
- **Training Data**: 150 synthetic Q&A pairs
- **Test Data**: 10 unseen questions

## Performance Metrics
| Metric   | Love | Career | Finance | Health |
|----------|-------|--------|---------|--------|
| Precision| 1.0   | 0.75   | 1.0     | 1.0    |
| Recall   | 1.0   | 1.0    | 0.5     | 1.0    |
| F1-Score | 1.0   | 0.86   | 0.67    | 1.0    |

**Overall Accuracy**: 90%

## Key Findings
1. **Strengths**:
   - Perfect classification for Love/Health questions
   - Handles career/finance ambiguity well
   - Fast prediction (<100ms per query)

2. **Weaknesses**:
   - Confuses Finance/Career questions (e.g., "Will I get a raise?")
   - Limited health vocabulary ("sick" vs "illness")

## Sample Outputs
1. **Input**: "When will I meet my soulmate?"  
   **Output**: "The Lovers card suggests a destined meeting within 6-12 months"

2. **Input**: "Should I quit my job?"  
   **Output**: "The Tower indicates sudden changes may be beneficial"

## Recommendations
1. Add more finance/career examples
2. Include slang terms ("broke", "job hop")
3. Implement rule-based fallbacks for edge cases
