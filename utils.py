import paddle

def read_json(path):
    english = []
    chinese = []
    print("start_read")
    with open(f'translation2019zh/translation2019zh_{path}.json', 'r', encoding='utf-8') as f:
        while True:
            try:
                line = eval(f.readline())
                english.append(line['english'])
                chinese.append(line['chinese'])
            except:
                break
    print("end_read")
    return english, chinese

def reshape_chinese_english_token(english, chinese):
    if len(english) == len(chinese):
        return english, chinese
    
    if len(english) < len(chinese):
        return paddle.nn.functional.pad(english, (0, len(chinese) - len(english), 0, 0)), chinese
    if len(english) > len(chinese):
        return english, paddle.nn.functional.pad(chinese, (0, len(english) - len(chinese), 0, 0))
    
if __name__ == "__main__":
    a = [[1, 2],
         [2, 3]]
    b = [[1, 2]]

    a, b = reshape_chinese_english_token(paddle.to_tensor(a, dtype='float32'), paddle.to_tensor(b, dtype='float32'))
    print(a, b)