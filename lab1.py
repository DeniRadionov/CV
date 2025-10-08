import numpy as np
import cv2 as cv
import glob
import yaml


CHESSBOARD = (7, 6)   # количество внутренних углов по горизонтали и вертикали на шахматной доске
SQUARE_SIZE = 1.0     # размер клетки (в любых единицах — только масштаб важен)
TERMINATION = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # критерии остановки уточнения углов
IMAGE_GLOB = '*.jpg'  # шаблон для поиска изображений с шахматкой
OUTPUT_YAML = 'calibration_result.yaml'  # файл для сохранения результатов калибровки

def make_object_points(board, square_size):
    # Генерация координат точек шахматной доски в 3D-пространстве 
    cols, rows = board
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


def draw_distortion_field(img, camera_matrix, dist_coeffs, step=40, scale=1.0):
    """
    Визуализирует, как искажения влияют на положение точек.
    Рисует сетку точек и показывает смещение от идеальных координат к искажённым.
    """
    h, w = img.shape[:2]
    # Создаём сетку из равномерных точек на изображении
    xs = np.arange(0, w, step)
    ys = np.arange(0, h, step)
    pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)

    # Преобразуем пиксельные координаты в нормализованные
    normalized = cv.undistortPoints(pts, camera_matrix, None, P=None)

    # Создаём 3D-точки для последующей проекции с применением искажений
    obj3d = np.hstack([normalized.reshape(-1, 2), np.ones((normalized.shape[0], 1))]).astype(np.float32)

    # Проецируем точки обратно в изображение с учётом дисторсии
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    projected, _ = cv.projectPoints(obj3d, rvec, tvec, camera_matrix, dist_coeffs)

    vis = img.copy()
    # Рисуем стрелки от исходных до искажённых позиций
    for p_orig, p_dist in zip(pts.reshape(-1, 2), projected.reshape(-1, 2)):
        p1 = tuple(np.round(p_orig).astype(int))
        p2 = tuple(np.round(p_dist).astype(int))
        cv.circle(vis, p1, 3, (0, 255, 0), -1)  # зелёная — исходная точка
        cv.circle(vis, p2, 3, (0, 0, 255), -1)  # красная — искажённая
        cv.line(vis, p1, p2, (255, 0, 0), 1)    # синяя линия — направление смещения
    return vis


def calibrate_and_visualize():
    # Подготовка шаблонных 3D-точек
    objp = make_object_points(CHESSBOARD, SQUARE_SIZE)

    objpoints = []  # 3D-точки в реальном пространстве
    imgpoints = []  # 2D-точки на изображениях
    images = glob.glob(IMAGE_GLOB)

    if len(images) == 0:
        raise RuntimeError(f'Не найдено изображений по шаблону: {IMAGE_GLOB}')

    last_gray = None
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print('Ошибка чтения файла', fname)
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        last_gray = gray

        # Поиск углов 
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            # Добавляем данные 
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERMINATION)
            imgpoints.append(corners2)

            # Отрисовка найденных углов
            cv.drawChessboardCorners(img, CHESSBOARD, corners2, ret)
            cv.imshow('Найденные углы', img)
            cv.waitKey(3000)
        else:
            print(f'Шахматная доска не найдена в {fname}')

    cv.destroyAllWindows()

    if len(objpoints) < 3:
        raise RuntimeError('Недостаточно изображений для калибровки (нужно >=3)')

    image_size = (last_gray.shape[1], last_gray.shape[0])

    # Калибровка камеры
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    # Вывод результатов
    print('RMS ошибка калибровки:', ret)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    skew = camera_matrix[0, 1]

    print(f'Матрица камеры:\n{camera_matrix}')
    print(f'Фокусные расстояния: fx={fx:.2f}, fy={fy:.2f}')
    print(f'Главная точка: cx={cx:.2f}, cy={cy:.2f}')
    print(f'Скошенность (skew): {skew:.5f}')
    print(f'Коэффициенты искажения (k1,k2,p1,p2,k3,...): {dist_coeffs.ravel()}')

    # Сохраняем результаты в YAML
    data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeff': dist_coeffs.tolist(),
        'rms': float(ret),
        'image_size': image_size,
    }
    with open(OUTPUT_YAML, 'w') as f:
        yaml.dump(data, f)
    print(f'Результаты сохранены в {OUTPUT_YAML}')

    # Пример коррекции искажений
    example_img = cv.imread(images[0])
    undistorted = cv.undistort(example_img, camera_matrix, dist_coeffs)
    cv.imshow('Оригинал', example_img)
    cv.imshow('Исправленное изображение', undistorted)

    # Визуализация поля искажений
    vis = draw_distortion_field(example_img, camera_matrix, dist_coeffs, step=40)
    cv.imshow('Поле искажений (зелёное=идеал, красное=искажённое, синее=смещение)', vis)

    print('\nНажмите любую клавишу...')
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    calibrate_and_visualize()
