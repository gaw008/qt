# Coding Rules and Communication Guidelines

## Established Rules (2025-09-16)

### 1. Code Language Requirement
- ALL code must be written in English only
- Variable names, function names, comments, and documentation in English
- No Chinese characters in any code files
- Error messages and logging output in English

### 2. No Icons or Emojis
- Do not use any icons, emojis, or Unicode symbols in code
- Avoid special characters that may cause encoding issues
- Use plain text only for better compatibility
- This prevents Unicode encoding errors in Windows environment

### 3. Communication Language
- Use Chinese for all conversations and discussions with user
- Code documentation and technical writing in English
- User interface and error messages in English for technical consistency
- Verbal communication and explanations in Chinese

## Implementation Guidelines

### Code Examples
```python
# CORRECT - English only
def calculate_position_size(stock_symbol, account_balance):
    """Calculate optimal position size for given stock"""
    return position_size

# INCORRECT - Chinese characters
def 计算仓位大小(股票代码, 账户余额):
    """计算给定股票的最优仓位大小"""
    return 仓位大小
```

### File Naming
- Use English for all file names
- Use underscores for separation: `stock_screener.py`
- No special characters or Chinese in file paths

### Comments and Documentation
```python
# CORRECT
# Calculate Kelly criterion position size
kelly_position = (win_rate * avg_win - loss_rate * avg_loss) / avg_win

# INCORRECT
# 计算凯利公式仓位大小
kelly_position = (胜率 * 平均盈利 - 败率 * 平均亏损) / 平均盈利
```

### Error Handling
```python
# CORRECT
try:
    result = process_data()
except Exception as e:
    logger.error(f"Data processing failed: {e}")

# INCORRECT
try:
    result = process_data()
except Exception as e:
    logger.error(f"数据处理失败: {e}")
```

## Rationale

### English Code Benefits
- Better compatibility with international systems
- Avoids encoding issues in different environments
- Standard practice in professional development
- Easier integration with external libraries

### No Icons/Emojis Benefits
- Prevents Unicode encoding errors
- Better terminal compatibility
- Professional appearance
- Consistent with enterprise standards

### Chinese Communication Benefits
- Natural communication with Chinese-speaking user
- Better understanding of complex concepts
- More efficient problem-solving discussions
- Comfortable user experience

## Compliance Verification
- All new code files must follow English-only rule
- Code reviews should check for Chinese characters
- No emojis or special Unicode symbols in any code
- Communication logs should be in Chinese for user interactions