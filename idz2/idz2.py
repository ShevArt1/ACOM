import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

IMAGE_FOLDER = 'idz2images/'

IMAGE_NAMES = ["first.jpg", "second.webp", "third.jpg"]

#GAUSSIAN_KERNEL_SIZES = [(3, 3), (5, 5), (7, 7)]
GAUSSIAN_SIGMA = [0.2, 1, 5]

# Пары пороговых значений для алгоритма Канни
CANNY_THRESHOLD_PAIRS = [
    (50, 150),
    (100, 200),
    (150, 250)
]

OPERATORS = {
    'Sobel': None,
    'Prewitt': None
}

# Альтернативные методы выявления границ
ALTERNATIVE_METHODS = {
    'Laplacian': None,
    # 'Zero_Crossing': None,
    # 'Difference_of_Gaussians': None
}
THRESHOLD = [10, 30, 50]

def load_images(folder, image_names):
    images = []
    valid_image_names = []
    for filename in image_names:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            valid_image_names.append(filename)
        else:
            print(f"Не удалось загрузить изображение: {filename}")
    return images, valid_image_names


def apply_canny(image, lower_thresh, upper_thresh, operator='Sobel'):
    if operator == 'Prewitt':
        # Ядра Прюитта
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        # Применение ядер
        grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

        # Вычисление абсолютных значений и объединение
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        edges = cv2.Canny(grad, lower_thresh, upper_thresh)
        return edges
    else:
        # Используем стандартный алгоритм Канни с оператором Собеля
        return cv2.Canny(image, lower_thresh, upper_thresh)


# Функция для применения альтернативных методов
def apply_alternative_method(image, method, thresh):
    if method == 'Laplacian':
        #laplacian = cv2.Laplacian(image, cv2.CV_64F)
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=np.float32)

        laplacian = cv2.filter2D(image, -1, kernel)
        edges = cv2.convertScaleAbs(laplacian)
        _, edges = cv2.threshold(edges, thresh, 255, cv2.THRESH_BINARY)
        return edges
    # elif method == 'Zero_Crossing':
    #     # Использование Zero Crossing на основе Лапласиана Гаусса
    #     blurred = cv2.GaussianBlur(image, (3,3), 0)
    #     laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    #     zero_cross = np.zeros_like(laplacian, dtype=np.uint8)
    #     # Определение нулевых переходов
    #     zero_cross[(laplacian > 0) & (cv2.Laplacian(blurred, cv2.CV_64F, ksize=3) < 0)] = 255
    #     zero_cross[(laplacian < 0) & (cv2.Laplacian(blurred, cv2.CV_64F, ksize=3) > 0)] = 255
    #     return zero_cross
    # elif method == 'Difference_of_Gaussians':
    #     # Разница Гауссианов
    #     blur1 = cv2.GaussianBlur(image, (3,3), 0)
    #     blur2 = cv2.GaussianBlur(image, (5,5), 0)
    #     dog = cv2.subtract(blur1, blur2)
    #     _, edges = cv2.threshold(dog, 10, 255, cv2.THRESH_BINARY)
    #     return edges
    else:
        return None

def display_comparison(original, results, image_name, save_path=None):
    num_methods = len(results)
    cols = 3  # Количество столбцов в сетке
    rows = (num_methods + 1) // cols + 1  # Количество строк, включая исходное изображение

    plt.figure(figsize=(5 * cols, 5 * rows))

    plt.subplot(rows, cols, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Исходное')
    plt.axis('off')

    for idx, result in enumerate(results):
        plt.subplot(rows, cols, idx + 2)
        plt.imshow(result['Edges'], cmap='gray')
        title = f"{result['Method']}"
        if result['Method'].startswith('Canny'):
            title += f"\nOp: {result['Operator']}\nSigma: {result['Gaussian Sigma']}\nThresh: {result['Lower Threshold']}-{result['Upper Threshold']}"
        elif result['Method'] in ['Zero_Crossing', 'Difference_of_Gaussians']:
            title += f"\nMethod: {result['Method']}"
        plt.title(title, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

images, image_names = load_images(IMAGE_FOLDER, IMAGE_NAMES)
print(f"Загружено изображений: {len(images)}.")

all_results = []
optimal_results = []

VISUALIZATION_FOLDER = 'visualizations/'
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Основной цикл обработки изображений
for idx, image in enumerate(images):
    image_name = image_names[idx]
    print(f"\nОбработка изображения: {image_name}")


    # Список для визуализации всех методов для текущего изображения
    comparison_results = []

    for operator_name, operator in OPERATORS.items():
        for sigma in GAUSSIAN_SIGMA:
            blurred = cv2.GaussianBlur(image, (7,7), sigma)
            for thresh_pair in CANNY_THRESHOLD_PAIRS:
                lower, upper = thresh_pair
                edges = apply_canny(blurred, lower, upper, operator=operator_name)

                comparison_results.append({
                    'Method': f'Canny ({operator_name})',
                    'Operator': operator_name,
                    'Gaussian Sigma': sigma,
                    'Lower Threshold': lower,
                    'Upper Threshold': upper,
                    'Edges': edges
                })
                cv2.imwrite(f'./idz2images/{image_name}{operator_name}{int(sigma)}_{upper}.jpg', edges)


    # Применение альтернативных методов
    for method_name in ALTERNATIVE_METHODS.keys():
        for thresh in THRESHOLD:
            edges = apply_alternative_method(image, method_name, thresh)
            comparison_results.append({
                'Method': method_name+ str(thresh),
                'Operator': 'N/A',
                'Gaussian Sigma': 'N/A',
                'Lower Threshold': 'N/A',
                'Upper Threshold': 'N/A',
                'Edges': edges
            })
            cv2.imwrite(f'./idz2images/{image_name}Laplasian{thresh}.jpg', edges)




    save_path = os.path.join(VISUALIZATION_FOLDER, f"{os.path.splitext(image_name)[0]}_comparison.png")
