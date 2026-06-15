import numpy as np
import time
import sys
from os import path
import pickle
import urllib.request
import tarfile
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

# ============================================
# ⚙️ ВСЕ НАСТРОЙКИ
# ============================================

# --- Вывод ---
DISPLAY_WARNINGS = False       # Выводить предупрежления Python? Не рекомендуется включать, если вы не программист
WRITE_TO_FILE = False                # Писать в консоль или в файл? Если True — в файл
FILE_PATH = "./results_01.txt"      # Путь к файлу, куда писать
ADD_MODE = False                        # Перезаписывать содержимое файла или добавлять? Если True — добавлять
# Если WRITE_TO_FILE установлена в False, FILE_PATH и ADD_MODE ничего не делают

# --- Данные ---
BATCH_COUNT = 2                        # Количество загружаемых из CIFAR-10 тренировочных батчей (в каждом 10000 примеров, их 5)
TRAIN_SIZE = 16000                     # Количество тренировочных примеров
TEST_SIZE = 16000                       # Количество тестовых примеров
FORCE_DOWNLOAD = False        # Принудительно скачивать обучающие данные заново? Если True — да

# --- Архитектура сети ---
HIDDEN_SIZES = [15] * 10            # Список: сколько нейронов в каждом скрытом слое

# --- Обучение ---
EPOCHS = 30                                 # Количество эпох
BATCH_SIZE = 100                        # Размер батча
LEARNING_RATE = 0.001             # Шаг градиентного спуска
LOG_EVERY = 10                           # Просчёт и вывод информации каждые N эпох

# --- Что сравниваем ---
COMPARE_WITH_SWISH = True  # Сравнивать со Swish? Если False — только LLA

# --- Рандом ---
DETERMINATION = True               # Сид рандома фиксирован? Если False — нет
RANDOM_SEED = 42                     # Сид рандома
# Если DETERMINATION установлена в False, RANDOM_SEED ничего не делает

# --- Инициализация ---
α = 0                                                # Первый коэффициент инициализации (init = √(α + β / pred) + γ), α
β = 0.875                                         # Второй коэффициент инициализации (init = √(α + β / pred) + γ), β, для LLA хорошо 0.875
γ = 0                                                # Третий коэффициент инициализации (init = √(α + β / pred) + γ), γ

# ============================================
# ✨ ФУНКЦИИ АКТИВАЦИИ
# ============================================

def lla(x):
    return x + np.log(np.abs(x) + 1.0)

def lla_derivative(x):
    return 1.0 + 1.0 / (np.abs(x) + 1.0) # Производная неправильная, но так и должно быть

def swish(x):
    return x / (1.0 + np.exp(-x))

def swish_derivative(x):
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig + x * sig * (1.0 - sig)

# ============================================
# 📖 ПОДГОТОВКА ДАННЫХ
# ============================================

if not DISPLAY_WARNINGS:
    warnings.filterwarnings('ignore')

if WRITE_TO_FILE:
    sys.stdout = open(FILE_PATH, "a" if ADD_MODE else "w")

def download_and_extract_cifar10(force):
    filename = "cifar-10-python.tar.gz"
    if not path.exists(filename) or force:
        print("Скачивание CIFAR-10 (первый раз, ~162MB)...")
        urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", filename)
    if not path.exists("cifar-10-batches-py") or force:
        print("Распаковка...")
        try:
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(filter='data')
        except TypeError:
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall()

def load_cifar10():
    download_and_extract_cifar10(FORCE_DOWNLOAD)
    X_list = []
    y_list = []
    for i in range(BATCH_COUNT):
        with open(f'cifar-10-batches-py/data_batch_{i + 1}', 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            X_list.append(batch[b'data'])
            y_list.append(batch[b'labels'])
    X = np.concatenate(X_list)[:TRAIN_SIZE]
    y = np.concatenate(y_list)[:TRAIN_SIZE]
    with open('cifar-10-batches-py/test_batch', 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        X_test = batch[b'data'][:TEST_SIZE]
        y_test = np.array(batch[b'labels'])[:TEST_SIZE] 
    return X, y, X_test, y_test

print("Загрузка данных...")
if DETERMINATION:
    np.random.seed(RANDOM_SEED)
X_raw, y_train, X_test_raw, y_test = load_cifar10()
X_raw = X_raw / 255.0
X_test_raw = X_test_raw / 255.0
scaler = StandardScaler()
X_train = scaler.fit_transform(X_raw)
X_test = scaler.transform(X_test_raw)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================
# 🧠 СЕТЬ
# ============================================

class FlexibleNN:
    def __init__(self, input_size, hidden_sizes, output_size, 
                 activation_func, activation_deriv, lr):
        self.layers = []
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv
        self.lr = lr
        
        prev = input_size
        for i, h in enumerate(hidden_sizes):
            W = np.random.randn(prev, h) * (np.sqrt(α + β / prev) + γ)
            b = np.zeros((1, h))
            self.layers.append({'W': W, 'b': b})
            prev = h
        
        self.W_out = np.random.randn(prev, output_size) * (np.sqrt(α + β / prev) + γ)
        self.b_out = np.zeros((1, output_size))
    
    def forward(self, X):
        self.cache = [X]
        for layer in self.layers:
            Z = self.cache[-1] @ layer['W'] + layer['b']
            A = self.activation_func(Z)
            self.cache.append(Z)
            self.cache.append(A)
        
        Z_out = self.cache[-1] @ self.W_out + self.b_out
        exp_z = np.exp(Z_out - np.max(Z_out, axis=1, keepdims=True))
        self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.output
    
    def backward(self, X, y_true):
        n = X.shape[0]
        dZ = self.output.copy()
        dZ[range(n), y_true] -= 1
        dZ /= n
        
        dW_out = self.cache[-1].T @ dZ
        db_out = np.sum(dZ, axis=0, keepdims=True)
        
        dA = dZ @ self.W_out.T
        grads = []
        
        for i in range(len(self.layers)-1, -1, -1):
            Z = self.cache[2*i + 1]
            dZ = dA * self.activation_deriv(Z)
            dW = self.cache[2*i].T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)
            grads.insert(0, (dW, db))
            dA = dZ @ self.layers[i]['W'].T
        
        self.W_out -= self.lr * dW_out
        self.b_out -= self.lr * db_out
        for i, (dW, db) in enumerate(grads):
            self.layers[i]['W'] -= self.lr * dW
            self.layers[i]['b'] -= self.lr * db
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# ============================================
# 👨‍🎓 ОБУЧЕНИЕ
# ============================================

def train_model(activation_func, activation_deriv, name):
    print(f"\n{'='*105}")
    print(f"📊 {name} | Слои: {HIDDEN_SIZES} | Эпохи: {EPOCHS} | Шаг градиентного спуска: {LEARNING_RATE}")
    print(f"{'='*105}")
    
    model = FlexibleNN(
        input_size=X_train.shape[1],
        hidden_sizes=HIDDEN_SIZES,
        output_size=10,
        activation_func=activation_func,
        activation_deriv=activation_deriv,
        lr=LEARNING_RATE
    )
    
    n = X_train.shape[0]
    best_acc = 0.0
    start = time.time()
    
    for epoch in range(EPOCHS):
        perm = np.random.permutation(n)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        for i in range(0, n, BATCH_SIZE):
            X_batch = X_shuffled[i:i+BATCH_SIZE]
            y_batch = y_shuffled[i:i+BATCH_SIZE]
            model.forward(X_batch)
            model.backward(X_batch, y_batch)
        
        if epoch % LOG_EVERY == 0 or epoch == EPOCHS-1:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch:5d}/{EPOCHS} | Test Acc: {acc:.4f}")
    
    elapsed = time.time() - start
    print(f"✅ Лучшая точность: {best_acc:.4f} | Время: {elapsed:.2f} секунд ({elapsed/60:.2f} минут(ы))")
    return best_acc

# ============================================
# ✅ СТАРТ
# ============================================

def russian(a, b, c, count):
    remainder = count % 100
    if remainder < 10 or remainder > 20:
        if remainder % 10 == 1:
            return a
        if remainder % 10 in {2, 3, 4}:
            return b
        else:
            return c
    else:
        return c 

def format_architecture(SIZES):
    if not SIZES:
        return "без скрытых слоёв"
    text = ""
    last_size = 0
    mul = 1
    for size in SIZES:
        if size == last_size:
            mul += 1
        else:
            if last_size != 0:
                text += f"{mul} сло{russian('й', 'я', 'ёв', mul)} по {last_size} нейрон{russian('', 'а', 'ов', last_size)}"
                text += " → "
            mul = 1
            last_size = size
    text += f"{mul} сло{russian('й', 'я', 'ёв', mul)} по {last_size} нейрон{russian('', 'а', 'ов', last_size)}"
    return text

print("\n" + "="*105)
print(f"КОНФИГУРАЦИЯ: {TRAIN_SIZE} трейн, {TEST_SIZE} тест, {BATCH_COUNT} батч{russian('', 'а', 'ей', BATCH_COUNT)} CIFAR-10")
print(f"СКРЫТЫЕ СЛОИ: {format_architecture(HIDDEN_SIZES)}")
print(f"{EPOCHS} ЭПОХ{russian('а', 'и', '', EPOCHS)}, LR: {LEARNING_RATE}, РАЗМЕР БАТЧЕЙ: {BATCH_SIZE}")
print(f"ИНИЦИАЛИЗАЦИЯ: √({β} / prev{'' if α == 0 else f' + {α}' if α > 0 else f' - {-α}'}){'' if γ == 0 else f' + {γ}' if γ > 0 else f' - {-γ}'}")
print("="*105)

acc_lla = train_model(lla, lla_derivative, "LLA")

if COMPARE_WITH_SWISH:
    acc_swish = train_model(swish, swish_derivative, "Swish")
    print("\n" + "="*105)
    print("ИТОГ:")
    print(f"• LLA:           {acc_lla:.4f} ({acc_lla*100:.2f}%)")
    print(f"• Swish:         {acc_swish:.4f} ({acc_swish*100:.2f}%)")
    print(f"• Разница:      {acc_lla - acc_swish:+.4f} ({(acc_lla - acc_swish) * 100:+.2f}%)")
    if acc_lla > acc_swish:
        print(f"🎉 LLA лучше swish в {acc_lla / acc_swish:.3f} раз(а)!")
    elif acc_lla == acc_swish:
        print("✨ Ничья! LLA и Swish показали идентичный результат.")
    else:
        print("🙂 Swish пока впереди.")
else:
    print(f"\n✅ LLA показала: {acc_lla:.4f} ({acc_lla*100:.2f}%)")
