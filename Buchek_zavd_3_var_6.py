import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FourPetalRose:
    def __init__(self, a_param=5.0, num_points=1000):
        if not isinstance(a_param, (int, float)) or a_param <= 0:
            raise ValueError("Параметр 'a' повинен бути позитивним числом.")
        self._a_param = a_param
        self._num_points = num_points
        self._initial_coords = self._generate_initial_coordinates()

    @property
    def a_param(self):
        return self._a_param

    @a_param.setter
    def a_param(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("Параметр 'a' повинен бути позитивним числом.")
        self._a_param = value
        self._initial_coords = self._generate_initial_coordinates()

    def _generate_initial_coordinates(self):
        phi = np.linspace(0, 2 * np.pi, self._num_points)
        r = self._a_param * np.sin(2 * phi)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.vstack((x, y)).T

    def get_initial_coordinates(self):
        return self._initial_coords

class AffineTransformedRose(FourPetalRose):
    def __init__(self, a_param=5.0, num_points=1000):
        super().__init__(a_param, num_points)
        self._transformed_coords = self._initial_coords.copy()

    def get_current_coordinates(self):
        return self._transformed_coords

    def reset_transformation(self):
        self._transformed_coords = self._initial_coords.copy()

    def _apply_transform_matrix(self, matrix):
        homogeneous_coords = np.hstack((self._transformed_coords, np.ones((self._transformed_coords.shape[0], 1))))
        transformed_homogeneous_coords = homogeneous_coords @ matrix.T
        self._transformed_coords = transformed_homogeneous_coords[:, :2]

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            scale_factor = other
            self._scale_about_center(scale_factor)
            return self

        elif isinstance(other, tuple) and len(other) == 2 and isinstance(other[0], str):
            transform_type = other[0]
            param = other[1]

            if transform_type == 'scale_area':
                if not isinstance(param, (int, float)) or param <= 0:
                    raise ValueError("Параметр 'площа' повинен бути позитивним числом.")
                self._scale_by_area_ratio(param)
                return self

            elif transform_type == 'scale_bbox_edge':
                if not isinstance(param, (int, float)) or param <= 0:
                    raise ValueError("Параметр 'довжина ребра' повинен бути позитивним числом.")
                self._scale_to_bbox_edge(param)
                return self

            elif transform_type == 'scale_point':
                if not (isinstance(param, tuple) and len(param) == 3 and isinstance(param[0], (int, float)) and isinstance(param[1], (int, float)) and isinstance(param[2], (int, float))):
                    raise ValueError("Параметр 'scale_point' повинен бути кортежем (x_point, y_point, scale_factor).")
                x_p, y_p, s_f = param
                self._scale_about_point(x_p, y_p, s_f)
                return self

            elif transform_type == 'rotate_center':
                if not isinstance(param, (int, float)):
                    raise ValueError("Параметр 'кут повороту' повинен бути числом.")
                self._rotate_about_center(np.deg2rad(param))
                return self
            
            elif transform_type == 'reflect_line':
                if not (isinstance(param, tuple) and len(param) == 4 and all(isinstance(val, (int, float)) for val in param)):
                    raise ValueError("Параметр 'reflect_line' повинен бути кортежем (x1, y1, x2, y2).")
                x1, y1, x2, y2 = param
                self._reflect_about_line(x1, y1, x2, y2)
                return self

        raise TypeError(f"Непідтримуваний тип операції для *=: {type(other)}")

    def _get_center(self):
        min_x, min_y = np.min(self._transformed_coords, axis=0)
        max_x, max_y = np.max(self._transformed_coords, axis=0)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        return np.array([center_x, center_y])

    def _scale_about_center(self, scale_factor):
        center = self._get_center()
        translation_matrix_to_origin = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-center[0], -center[1], 1]
        ])
        scaling_matrix = np.array([
            [scale_factor, 0, 0],
            [0, scale_factor, 0],
            [0, 0, 1]
        ])
        translation_matrix_back = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [center[0], center[1], 1]
        ])
        transform_matrix = translation_matrix_to_origin @ scaling_matrix @ translation_matrix_back
        self._apply_transform_matrix(transform_matrix)

    def _get_current_area(self):
        x, y = self._transformed_coords[:,0], self._transformed_coords[:,1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _scale_by_area_ratio(self, target_area_ratio):
        current_area = self._get_current_area()
        if current_area == 0:
            messagebox.showwarning("Помилка", "Поточна площа фігури нульова, масштабування за площею неможливе.")
            return
        scale_factor = np.sqrt(target_area_ratio)
        self._scale_about_center(scale_factor)

    def _scale_to_bbox_edge(self, target_edge_length):
        min_x, min_y = np.min(self._transformed_coords, axis=0)
        max_x, max_y = np.max(self._transformed_coords, axis=0)
        
        current_width = max_x - min_x
        current_height = max_y - min_y

        if current_width == 0 or current_height == 0:
            messagebox.showwarning("Помилка", "Ширина або висота фігури нульова, масштабування за розміром ребра неможливе.")
            return

        scale_factor_x = target_edge_length / current_width
        scale_factor_y = target_edge_length / current_height
        scale_factor = min(scale_factor_x, scale_factor_y)
        
        self._scale_about_center(scale_factor)
        
    def _scale_about_point(self, center_x, center_y, scale_factor):
        point = np.array([center_x, center_y])
        
        translation_to_point = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-point[0], -point[1], 1]
        ])
        scaling_matrix = np.array([
            [scale_factor, 0, 0],
            [0, scale_factor, 0],
            [0, 0, 1]
        ])
        translation_back_from_point = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [point[0], point[1], 1]
        ])
        
        transform_matrix = translation_to_point @ scaling_matrix @ translation_back_from_point
        self._apply_transform_matrix(transform_matrix)

    def _rotate_about_center(self, angle_rad):
        center = self._get_center()
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,      1]
        ])
        
        translation_to_origin = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-center[0], -center[1], 1]
        ])
        translation_back = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [center[0], center[1], 1]
        ])
        
        transform_matrix = translation_to_origin @ rotation_matrix @ translation_back
        self._apply_transform_matrix(transform_matrix)

    def _reflect_about_line(self, x1, y1, x2, y2):
        tx, ty = -x1, -y1
        T1 = np.array([[1, 0, 0], [0, 1, 0], [tx, ty, 1]])

        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            messagebox.showwarning("Помилка", "Точки, що визначають пряму, збігаються.")
            return
            
        angle = np.arctan2(dy, dx)
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        R1 = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        M = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

        cos_a_inv, sin_a_inv = np.cos(angle), np.sin(angle)
        R2 = np.array([[cos_a_inv, -sin_a_inv, 0], [sin_a_inv, cos_a_inv, 0], [0, 0, 1]])

        T2 = np.array([[1, 0, 0], [0, 1, 0], [-tx, -ty, 1]])

        transform_matrix = T1 @ R1 @ M @ R2 @ T2
        self._apply_transform_matrix(transform_matrix)


class RoseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Чотирипелюсткова троянда з перетвореннями")
        self.geometry("900x800")

        self.rose = AffineTransformedRose(a_param=5.0)

        self.create_menu()
        self.create_widgets()
        self.plot_rose()

    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Вихід", command=self.on_closing)

        options_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Параметри", menu=options_menu)
        options_menu.add_command(label="Ввести параметр 'a'", command=self.set_a_parameter)
        options_menu.add_command(label="Скинути перетворення", command=self.reset_transformation)

        transform_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Перетворення", menu=transform_menu)
        
        transform_menu.add_command(label="1. Масштабування за коеф. (к*=2)", command=lambda: self.apply_transformation('scale_factor'))
        transform_menu.add_command(label="2. Масштабування за площею (к*=співвідношення)", command=lambda: self.apply_transformation('scale_area'))
        transform_menu.add_command(label="3. Масштабування за ребром (к*=довжина)", command=lambda: self.apply_transformation('scale_bbox_edge'))
        transform_menu.add_command(label="4. Масштабування від точки (к*=(х,у,коеф))", command=lambda: self.apply_transformation('scale_point'))
        transform_menu.add_command(label="5. Поворот від центру (к*=('rotate', кут_град))", command=lambda: self.apply_transformation('rotate_center'))
        transform_menu.add_command(label="6. Дзеркальне відображення (к*=('reflect', x1,y1,x2,y2))", command=lambda: self.apply_transformation('reflect_line'))

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Довідка", menu=help_menu)
        help_menu.add_command(label="Про програму", command=self.show_about_info)

    def create_widgets(self):
        control_frame = tk.Frame(self, bd=2, relief=tk.RAISED)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.a_label = tk.Label(control_frame, text=f"Поточний параметр 'a': {self.rose.a_param:.2f}")
        self.a_label.pack(side=tk.LEFT, padx=5)

        plot_button = tk.Button(control_frame, text="Скинути перетворення", command=self.reset_and_plot_rose)
        plot_button.pack(side=tk.RIGHT, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def reset_and_plot_rose(self):
        self.rose.reset_transformation()
        self.plot_rose()
        messagebox.showinfo("Графік оновлено", "Усі перетворення скинуто до початкових.")

    def set_a_parameter(self):
        while True:
            try:
                new_a = simpledialog.askfloat("Введення параметра", "Введіть значення параметра 'a':",
                                              initialvalue=self.rose.a_param)
                if new_a is None:
                    return
                self.rose.a_param = new_a
                self.rose.reset_transformation()
                self.a_label.config(text=f"Поточний параметр 'a': {self.rose.a_param:.2f}")
                self.plot_rose()
                break
            except ValueError as e:
                messagebox.showerror("Помилка введення", str(e) + "\nБудь ласка, введіть позитивне число.")
            except Exception as e:
                messagebox.showerror("Помилка", f"Виникла невідома помилка: {e}")
                return

    def reset_transformation(self):
        self.rose.reset_transformation()
        self.plot_rose()
        messagebox.showinfo("Скидання", "Усі перетворення скинуто.")

    def apply_transformation(self, transform_type):
        try:
            if transform_type == 'scale_factor':
                scale_factor = simpledialog.askfloat("Масштабування", "Введіть коефіцієнт масштабування:", initialvalue=1.5)
                if scale_factor is None: return
                self.rose *= scale_factor
                messagebox.showinfo("Перетворення", f"Застосовано масштабування з коефіцієнтом {scale_factor}.")
            
            elif transform_type == 'scale_area':
                area_ratio = simpledialog.askfloat("Масштабування за площею", "Введіть співвідношення площі (наприклад, 2 для збільшення вдвічі):", initialvalue=2.0)
                if area_ratio is None: return
                self.rose *= ('scale_area', area_ratio)
                messagebox.showinfo("Перетворення", f"Застосовано масштабування: площа змінена у {area_ratio} разів.")

            elif transform_type == 'scale_bbox_edge':
                edge_length = simpledialog.askfloat("Масштабування за ребром", "Введіть бажану довжину ребра для вписування:", initialvalue=10.0)
                if edge_length is None: return
                self.rose *= ('scale_bbox_edge', edge_length)
                messagebox.showinfo("Перетворення", f"Застосовано масштабування: вписано в квадрат з ребром {edge_length}.")

            elif transform_type == 'scale_point':
                res = simpledialog.askstring("Масштабування від точки", "Введіть x, y точки та коефіцієнт масштабування через кому (напр., 0,0,1.5):")
                if res is None: return
                parts = [float(p.strip()) for p in res.split(',')]
                if len(parts) != 3: raise ValueError("Некоректний формат вводу. Потрібно 3 числа.")
                x_p, y_p, s_f = parts
                self.rose *= ('scale_point', (x_p, y_p, s_f))
                messagebox.showinfo("Перетворення", f"Застосовано масштабування від точки ({x_p},{y_p}) з коеф. {s_f}.")

            elif transform_type == 'rotate_center':
                angle_deg = simpledialog.askfloat("Поворот", "Введіть кут повороту в градусах:", initialvalue=45.0)
                if angle_deg is None: return
                self.rose *= ('rotate_center', angle_deg)
                messagebox.showinfo("Перетворення", f"Застосовано поворот на {angle_deg}°.")

            elif transform_type == 'reflect_line':
                res = simpledialog.askstring("Дзеркальне відображення", "Введіть координати двох точок прямої (x1,y1,x2,y2) через кому (напр., 0,0,1,1):")
                if res is None: return
                parts = [float(p.strip()) for p in res.split(',')]
                if len(parts) != 4: raise ValueError("Некоректний формат вводу. Потрібно 4 числа.")
                x1, y1, x2, y2 = parts
                self.rose *= ('reflect_line', (x1, y1, x2, y2))
                messagebox.showinfo("Перетворення", f"Застосовано дзеркальне відображення відносно прямої ({x1},{y1})-({x2},{y2}).")


            self.plot_rose()

        except ValueError as e:
            messagebox.showerror("Помилка вводу", str(e))
        except Exception as e:
            messagebox.showerror("Помилка перетворення", f"Виникла помилка: {e}")

    def plot_rose(self):
        self.ax.clear()

        current_coords = self.rose.get_current_coordinates()
        initial_coords = self.rose.get_initial_coordinates()

        self.ax.plot(initial_coords[:,0], initial_coords[:,1], color='gray', linestyle='--', linewidth=1, label='Початкова фігура')
        self.ax.plot(current_coords[:,0], current_coords[:,1], color='red', linewidth=2.5, label='Перетворена фігура')

        self.ax.set_title(f'Чотирипелюсткова троянда (a = {self.rose.a_param:.2f})')
        self.ax.set_xlabel('X-координата')
        self.ax.set_ylabel('Y-координата')
        self.ax.grid(True)
        self.ax.axhline(0, color='black', linewidth=0.8)
        plt.axvline(0, color='black', linewidth=0.8)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.legend()
        self.canvas.draw()

    def show_about_info(self):
        messagebox.showinfo("Про програму",
                            "Додаток для побудови чотирипелюсткової троянди з афінними перетвореннями.\n"
                            "Розроблено для навчальної дисципліни «Літня обчислювальна практика».\n"
                            "Завдання №1 та №3: Робота із графікою та ООП.\n"
                            "Автор: Бучек Іван 201")

    def on_closing(self):
        if messagebox.askokcancel("Вихід", "Ви впевнені, що хочете вийти?"):
            self.destroy()

if __name__ == "__main__":
    app = RoseApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()