import numpy as np
import scipy.optimize as opt
import scipy.signal as signal
import matplotlib.pyplot as plt
import time


def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def benchmark_optimization():
    methods = ['BFGS', 'CG', 'Nelder-Mead', 'Powell']
    results = {}

    for method in methods:
        start = time.time()

        result = opt.minimize(rosenbrock, x0=np.random.random(10) * 2, method=method)

        end = time.time()

        results[method] = {
            "time": end - start,
            "iterations": result.nit if hasattr(result, "nit") else None,
            "success": result.success,
            "minimum": result.fun,
        }

    return results

def generate_signal():
    fs = 1000  # частота дискретизации
    t = np.linspace(0, 2, 2 * fs, endpoint=False)
    signal_raw = 0.6 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)
    noise = 0.4 * np.random.normal(0, 1, len(t))
    return t, signal_raw + noise


def apply_fft(sig, fs=1000):
    freqs = np.fft.fftfreq(len(sig), 1 / fs)
    spectrum = np.abs(np.fft.fft(sig))
    return freqs, spectrum


def filter_signal(sig, fs=1000):
    b, a = signal.butter(4, 0.1)  # низкочастотный фильтр
    filtered = signal.filtfilt(b, a, sig)
    return filtered


def visualize_signal(t, raw_sig, filtered_sig, freqs, spectrum):
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 2, 1)
    plt.plot(t, raw_sig)
    plt.title("Исходный шумный сигнал")

    plt.subplot(2, 2, 2)
    plt.plot(freqs[:len(freqs)//2], spectrum[:len(freqs)//2])
    plt.title("Частотный спектр")

    plt.subplot(2, 1, 2)
    plt.plot(t, filtered_sig)
    plt.title("Отфильтрованный сигнал")

    plt.tight_layout()
    plt.show()


def main():
    print("Бенчмарк оптимизации:")
    results = benchmark_optimization()
    for method, res in results.items():
        print(method, res)

    print("\nОбработка сигнала...")

    t, raw = generate_signal()
    freqs, spectrum = apply_fft(raw)
    filtered = filter_signal(raw)

    visualize_signal(t, raw, filtered, freqs, spectrum)


if __name__ == "__main__":
    main()
