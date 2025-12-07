# $LAN=PYTHON$
# Author=徐士諭 (SHIH-YU SHU)
# Student ID=M143040012

# File=datastructure.py
# 編譯需要 python + pip install pandas
import pandas as pd

class WingedEdge:
    """
    索引：
    - 邊 (Edge) 相關列表使用 'k' 作為索引。
    - 頂點 (Vertex) 相關列表使用 'j' 作為索引。
    - 多邊形 (Polygon) 相關列表使用 'i' 作為索引。
    """
    def __init__(self):
        # --- 8 個 "邊" 陣列 (indexed by k) ---
        self.left_polygon = []
        self.right_polygon = []
        self.start_vertex = []
        self.end_vertex = []
        self.cw_predecessor = []
        self.ccw_predecessor = []
        self.cw_successor = []
        self.ccw_successor = []

        # --- 2 個 "查找" 陣列 ---
        self.edge_around_polygon = [] # (indexed by i) 這個列表儲存的是多邊形 i 邊界上的「任意一條邊」的 ID k
        self.edge_around_vertex = []  # (indexed by j) 這個列表儲存的是與頂點 j 相鄰的「任意一條邊」的 ID k

        # --- 3 個 "頂點" 陣列 (indexed by j) ---
        self.w_vertex = []  # 1 = 普通點, 0 = 無限遠點
        self.x_vertex = []  # x 座標
        self.y_vertex = []  # y 座標

        self.site_of_polygon = []  # polygon i 對應哪一個 site index

    def add_vertex(self, x, y, w=1):
        """
        添加一個新頂點並返回其 ID (索引)。
        
        Args:
            x (float): x 座標
            y (float): y 座標
            w (int): 頂點類型. 1 = 普通, 0 = 無限遠. 預設為 1.

        Returns:
            int: 新頂點的 ID (j)。
        """
        self.w_vertex.append(w)
        self.x_vertex.append(x)
        self.y_vertex.append(y)
        
        # 為 edge_around_vertex 添加一個預留位置
        self.edge_around_vertex.append(None)
        
        # 返回新頂點的索引 (ID)
        return len(self.w_vertex) - 1

    def add_polygon(self, site_index=None):
        self.edge_around_polygon.append(None)
        self.site_of_polygon.append(site_index)
        return len(self.edge_around_polygon) - 1

    def add_edge(self, start_v, end_v, left_p, right_p, 
                 cw_pred, ccw_pred, cw_succ, ccw_succ):
        """
        添加一個新邊並返回其 ID (索引)。

        Args:
            start_v (int): start.vertex[k] (頂點 j)
            end_v (int): end.vertex[k] (頂點 j)
            left_p (int): left.polygon[k] (多邊形 i)
            right_p (int): right.polygon[k] (多邊形 i)
            cw_pred (int): cw.predecessor[k] (邊 k)
            ccw_pred (int): ccw.predecessor[k] (邊 k)
            cw_succ (int): cw.successor[k] (邊 k)
            ccw_succ (int): ccw.successor[k] (邊 k)

        Returns:
            int: 新邊的 ID (k)。
        """
        self.start_vertex.append(start_v)
        self.end_vertex.append(end_v)
        self.left_polygon.append(left_p)
        self.right_polygon.append(right_p)
        self.cw_predecessor.append(cw_pred)
        self.ccw_predecessor.append(ccw_pred)
        self.cw_successor.append(cw_succ)
        self.ccw_successor.append(ccw_succ)
        return len(self.start_vertex) - 1

    def set_edge_around_vertex(self, vertex_id, edge_id):
        """
        設置 'edge.around.vertex[j]' 查找連結。
        """
        self.edge_around_vertex[vertex_id] = edge_id

    def set_edge_around_polygon(self, polygon_id, edge_id):
        """
        設置 'edge.around.polygon[i]' 查找連結。
        """
        self.edge_around_polygon[polygon_id] = edge_id

    def edges_and_polygon_around_vertex(self, vj):
        """
        給頂點 j，繞著它把 incident edges 和 polygons 依逆時針 (CCW) 列出。
        """
        k_start = self.edge_around_vertex[vj]
        if k_start is None:
            return [], []

        L_e_s = set()
        L_p = []

        k = k_start
        
        # 安全計數器，防止死循環
        for _ in range(1000):
            L_e_s.add(k)

            last_k = k
            
            # 判斷 vj 是這條邊 k 的起點還是終點
            if self.start_vertex[k] == vj:
                L_p.append(self.left_polygon[k])
                
                # 轉動到下一條邊
                k = self.ccw_predecessor[k]
                
            else:
                L_p.append(self.right_polygon[k])
                
                # 轉動到下一條邊
                k = self.cw_successor[k]

            # 終止條件
            if k is None or k == k_start or k == -1 or k == last_k:
                break

        k = k_start
        for _ in range(1000):
            L_e_s.add(k)

            last_k = k
            
            # 判斷 vj 是這條邊 k 的起點還是終點
            if self.start_vertex[k] == vj:
                L_p.append(self.left_polygon[k])
                
                # 轉動到下一條邊
                k = self.cw_predecessor[k]
                
            else:
                L_p.append(self.right_polygon[k])
                
                # 轉動到下一條邊
                k = self.ccw_successor[k]

            # 終止條件
            if k is None or k == k_start or k == -1 or k == last_k:
                break
        
        return list(L_e_s), L_p

    # 方便看目前有幾個東西
    @property
    def num_edges(self):
        return len(self.start_vertex)

    @property
    def num_vertices(self):
        return len(self.x_vertex)

    @property
    def num_polygons(self):
        return len(self.edge_around_polygon)

    # Debug：把 edge 資訊丟進 DataFrame
    def edges_dataframe(self):
        data = {
            "k": list(range(self.num_edges)),
            "left_polygon": self.left_polygon,
            "right_polygon": self.right_polygon,
            "start_vertex": self.start_vertex,
            "end_vertex": self.end_vertex,
            "cw_pred": self.cw_predecessor,
            "ccw_pred": self.ccw_predecessor,
            "cw_succ": self.cw_successor,
            "ccw_succ": self.ccw_successor,
        }
        return pd.DataFrame(data)
    
    def vertices_dataframe(self):
        data = {
            "j": list(range(self.num_vertices)),
            "w": self.w_vertex,
            "x": self.x_vertex,
            "y": self.y_vertex,
            "edge_around_vertex": self.edge_around_vertex,
        }
        return pd.DataFrame(data)

    def polygons_dataframe(self):
        data = {
            "i": list(range(self.num_polygons)),
            "edge_around_polygon": self.edge_around_polygon,
        }
        return pd.DataFrame(data)

# =================================================================================================================================================#

# $LAN=PYTHON$
# Author=徐士諭 (SHIH-YU SHU)
# Student ID=M143040012

# File=main.py
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import math
import copy

DEBUG = False

def is_left(start, end, point):
    # Cross product: (end - start) x (point - start)
    # return (end.x - start.x)*(point.y - start.y) - (end.y - start.y)*(point.x - start.x)
    val = (end[0] - start[0]) * (point[1] - start[1]) - \
            (end[1] - start[1]) * (point[0] - start[0])
    return val > 0

class VoronoiDiagram:
    def __init__(self, root):
        self.root = root
        self.root.title("Voronoi 初測")

        self.width = 600
        self.height = 600

        # 儲存點擊座標的列表
        self.points = []
        self.point_index = {}
        # 儲存線段座標的列表
        self.lines = []

        self.voronoi = WingedEdge()


        # TODO
        self.history = [] 
        self.current_step_index = -1

        # 儲存從 "測試腳本" 讀取的所有案例
        self.test_cases = []
        self.current_test_index = -1

        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg='white', relief="solid", borderwidth=1)
        self.canvas.pack(pady=10, padx=10)

        self.canvas.bind("<Button-1>", self.on_mouse_click)

        # 建立一個框架 (Frame) 來放置按鈕
        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(pady=5)

        self.clear_button = ttk.Button(self.button_frame, text="清空/重置", command=self.reset)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # 分隔線
        ttk.Separator(self.button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 讀取 "測試腳本" 按鈕
        self.load_button = ttk.Button(self.button_frame, text="讀取檔案", command=self.load_test_script_file)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.next_test_button = ttk.Button(self.button_frame, text="執行下一組", command=self.run_next_test)
        self.next_test_button.pack(side=tk.LEFT, padx=5)

        self.step_button = ttk.Button(self.button_frame, text="Step by Step", command=self.on_step_click)
        self.step_button.pack(side=tk.LEFT, padx=5)

        self.run_button = ttk.Button(self.button_frame, text="Run (Finish)", command=self.on_run_click)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(self.button_frame, text="儲存檔案", command=self.save_file)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # 讀取 "儲存檔" (P/E 格式) 按鈕
        self.load_saved_button = ttk.Button(self.button_frame, text="讀取(儲存檔)", command=self.load_saved_voronoi_file)
        self.load_saved_button.pack(side=tk.LEFT, padx=5)

    def on_step_click(self):
        if not self.points: return
        
        if len(self.points) <= 3:
            self.draw_voronoi()
            return

        self.canvas.delete('voronoi')
        if not self.history:
            print("--- 開始計算並錄製步驟 ---")
            # 確保 points 有排序
            sorted_points = sorted(self.points, key=lambda p: (p[0], p[1]))
            
            # 清空舊紀錄
            self.history = [] 
            self.current_step_index = -1
            
            # 執行演算法，這會填滿 self.history
            self.voronoi, _ = self.build_voronoi_dc(sorted_points)
            
            print(f"計算完成，共產生 {len(self.history)} 個步驟快照")
        if self.current_step_index < len(self.history) - 1:
            self.current_step_index += 1

            snapshot = self.history[self.current_step_index]
            self.step_by_step(snapshot)
            
            # 更新標題或狀態列顯示進度
            print(f"Step {self.current_step_index + 1}/{len(self.history)}: {snapshot['type']}")

            if self.current_step_index >= len(self.history) - 1:
                self.current_step_index = -1

    def reset_step(self):
        self.history = []
        self.current_step_index = -1
        self.clear_canvas_step()

    def on_run_click(self):
        self.reset_step()
        self.draw_voronoi()

    def on_mouse_click(self, event):
        x, y = event.x, event.y

        p = (x, y)

        if p in self.point_index:
            print(f"警告：點 ({x}, {y}) 已存在，忽略此次點擊。")
            return
        
        idx = len(self.points)
        self.points.append(p)
        self.point_index[p] = idx

        # self.points.append((x, y))
        print(f"新增點: ({x}, {y})，目前共 {len(self.points)} 點")

        self.draw_point(x, y)

        self.reset_step()

        # self.draw_voronoi()

    def load_saved_voronoi_file(self):
        """
        - P x y 和 E x1 y1 x2 y2 格式
        """
        file_path = filedialog.askopenfilename(
            title="選擇輸入的文字檔案",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not file_path:
            return

        print(f"--- 正在讀取檔案: {file_path} ---")
        
        # 讀檔前，先清空畫布
        self.reset()

        # 標記是否在檔案中讀到了線
        # 如果讀到了線，就不在結尾呼叫 draw_voronoi()
        found_lines = False

        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    clean_line = line.strip()

                    # 註解直接跳過
                    if clean_line.startswith("#") or not clean_line:
                        continue 

                    parts = clean_line.split()
                    if not parts:
                        continue

                    try:
                        if parts[0] == 'P':
                            # 格式: P x y
                            x = float(parts[1])
                            y = float(parts[2])
                            
                            self.points.append((x, y))
                            self.draw_point(x, y)

                        elif parts[0] == 'E':
                            # 格式: E x1 y1 x2 y2
                            # 允許座標超出視窗大小
                            x1 = float(parts[1])
                            y1 = float(parts[2])
                            x2 = float(parts[3])
                            y2 = float(parts[4])
                            
                            # 儲存並繪製
                            p1 = (x1, y1)
                            p2 = (x2, y2)
                            self.lines.append((p1, p2))
                            self.canvas.create_line(x1, y1, x2, y2, fill="blue", tags="voronoi")
                            found_lines = True # 標記讀到線了
                        else:
                            x = float(parts[0])
                            y = float(parts[1])
                            self.points.append((x, y))
                            self.draw_point(x, y)

                    except (IndexError, ValueError):
                        print(f"警告：忽略格式錯誤的行: '{line.strip()}'")

            print(f"檔案讀取完成，共載入 {len(self.points)} 個點, {len(self.lines)} 條線")
            self.point_index = {p: i for i, p in enumerate(self.points)}
            
            if not found_lines:
                print("未在檔案中偵測到線條，開始計算 Voronoi...")
                self.draw_voronoi()
            else:
                print("已從檔案載入 Voronoi 線條。")

        except FileNotFoundError:
            print(f"錯誤：找不到檔案 {file_path}")
        except Exception as e:
            print(f"讀取檔案時發生錯誤: {e}")

    def load_test_script_file(self):
        """
        讀取 n... (x,y) 格式的 "測試腳本"
        """
        file_path = filedialog.askopenfilename(
            title="選擇測試腳本檔案",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not file_path:
            return

        print(f"--- 正在解析測試腳本: {file_path} ---")

        # 重置測試案例
        self.test_cases = []
        self.current_test_index = -1
        
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines_iterator = iter(f)
                
                while True:
                    line = ""
                    try:
                        # 尋找 n
                        line = next(lines_iterator).strip()
                        
                        # 跳過註解和空行
                        while line.startswith("#") or not line:
                            line = next(lines_iterator).strip()

                        # 讀到 n
                        n = int(line)
                        if n == 0:
                            print("讀入點數為零，檔案測試停止。")
                            break
                        
                        current_case_points = []
                        print(f"讀取 n={n} 的案例...")
                        for _ in range(n):
                            point_line = next(lines_iterator).strip()
                            
                            while point_line.startswith("#") or not point_line:
                                point_line = next(lines_iterator).strip()
                                
                            parts = point_line.split()
                            x = float(parts[0])
                            y = float(parts[1])
                            if (x, y) in current_case_points:
                                print(f"警告：案例中包含重複點 ({x}, {y})，將忽略此點。")
                            else:
                                current_case_points.append((x, y))
                        
                        self.test_cases.append(current_case_points)

                    except StopIteration:
                        # 檔案正常讀完
                        break
                    except (IndexError, ValueError):
                        print(f"警告：忽略格式錯誤的行: '{line}'")

            print(self.test_cases)
            print(f"測試腳本解析完成，共載入 {len(self.test_cases)} 組測試案例。")
            if self.test_cases:
                print("請點擊 [執行下一組] 開始測試。")
        
        except FileNotFoundError:
            print(f"錯誤：找不到檔案 {file_path}")
        except Exception as e:
            print(f"讀取腳本時發生錯誤: {e}")

    def run_next_test(self):
        self.current_test_index += 1
        
        if self.current_test_index >= len(self.test_cases):
            print("--- 所有測試案例已執行完畢 ---")
            self.current_test_index = len(self.test_cases) - 1 # 停在最後
            return

        # 執行測試
        print(f"--- 執行測試 {self.current_test_index + 1} / {len(self.test_cases)} ---")
        
        self.reset()
        
        # 一次載入一組測試資料
        points_to_draw = self.test_cases[self.current_test_index]
        
        for p in points_to_draw:
            self.points.append(p)
            self.draw_point(p[0], p[1])
        self.point_index = {p: i for i, p in enumerate(self.points)}
        print(f"載入 {len(self.points)} 個點 : {self.points}。")
        
        # 演算法 draw_voronoi 內部已經有 n=1, 2, 3 的判斷
        # self.draw_voronoi()

    def save_file(self):
        file_path = filedialog.asksaveasfilename(
            title="儲存檔案",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            defaultextension=".txt"
        )
        
        if not file_path:
            return

        print(f"--- 正在儲存檔案: {file_path} ---")

        try:
            with open(file_path, 'w') as f:
                # sorted() 預設會先依 x 再依 y 排序
                sorted_points = sorted(self.points)
                
                f.write("# --- Points ---\n")
                for p in sorted_points:
                    f.write(f"P {int(p[0])} {int(p[1])}\n")

                canonical_lines = []  # self.lines 裡面的 P1, P2 可能順序不同，雖然是同樣的線但還是要先處理成統一格式
                for p1, p2 in self.lines:
                    if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] > p2[1]):
                        p1, p2 = p2, p1  # 交換
                    canonical_lines.append((p1, p2))
                
                # 預設會先依 p1 排序，再依 p2 排序
                sorted_lines = sorted(canonical_lines)

                f.write("# --- Lines ---\n")
                for (x1, y1), (x2, y2) in sorted_lines:
                    # 格式 'E x1 y1 x2 y2'
                    # 這邊座標要在self.width和self.height範圍
                    
                    f.write(f"E {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")

            print(f"儲存完畢. 共 {len(sorted_points)} 點, {len(sorted_lines)} 條線.")

        except Exception as e:
            print(f"儲存檔案時發生錯誤: {e}")

    def draw_point(self, x, y, color='black', tag='point', size=2):
        self.canvas.create_oval(x - size, y - size, x + size, y + size, fill=color, outline=color, tags=tag)


        if DEBUG:
            self.canvas.create_text(
                x + 8, y - 8,                   # 文字位置，稍微偏離點，不要蓋到
                text=f"({x:.0f}, {y:.0f})",     # 這裡可以依需要調整小數位
                anchor="sw",
                fill=color,
                tags=tag
            )
            self.root.update_idletasks()
            self.root.update()

    def reset(self):
        """清空畫布與點列表"""
        print("--- reset ---")
        self.canvas.delete("all")
        self.points = []
        self.lines = []
        self.point_index = {}
        self.voronoi = WingedEdge()
        self.reset_step()

    def clear_canvas(self, tag='all'):
        self.canvas.delete(tag)

    def get_perpendicular_bisector(self, p1, p2):
        """
        計算 p1 和 p2 的垂直平分線
        返回 (slope, intercept) 或 (None, x_intercept) [垂直線] 或 (0, y_intercept) [水平線]
        """
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2

        if p1 == p2:
            print("警告：嘗試計算相同點的垂直平分線，返回 None。")
            return None

        # 處理 p1, p2 兩點 x 座標相同，垂直平分線就會是水平線
        if p1[0] == p2[0]:
            # 原線是垂直線，平分線是水平線
            slope = 0
            intercept = mid_y
            return (slope, intercept)

        # 處理 p1, p2 兩點 y 座標相同，垂直平分線就會是垂直線
        if p1[1] == p2[1]:
            # (None, x_intercept) 來表示垂直線 x = x_intercept
            slope = None
            intercept = mid_x
            return (slope, intercept)

        # 一般情況
        original_slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        
        # 這邊理論上不會發生，因為前面已經處理過水平和垂直線的情況
        # if original_slope == 0:
        #     # 如果原線是水平線，平分線就會是垂直線
        #     slope = None
        #     intercept = mid_x
        #     return (slope, intercept)

        bisector_slope = -1 / original_slope
        
        # y = mx + c  =>  c = y - mx
        bisector_intercept = mid_y - bisector_slope * mid_x
        
        return (bisector_slope, bisector_intercept)

    def get_line_intersection(self, line1, line2):
        slope1, intercept1 = line1
        slope2, intercept2 = line2

        # 兩線平行，垂直得先做
        if slope1 is None and slope2 is None:
            return None

        # line1 垂直 (x = intercept1)
        if slope1 is None:
            x = intercept1
            y = slope2 * x + intercept2  # 尋找 line2 在 x 上的 y
            return (x, y)

        # line2 垂直 (x = intercept2)
        if slope2 is None:
            x = intercept2
            y = slope1 * x + intercept1
            return (x, y)
        
        # 非垂直線平行
        if (abs(slope1 - slope2) < 1e-9):
            return None # 無交點

        # 一般情況 (m1*x + b1 = m2*x + b2)
        # (m1 - m2) * x = b2 - b1
        # x = (b2 - b1) / (m1 - m2)
        x = (intercept2 - intercept1) / (slope1 - slope2)
        y = slope1 * x + intercept1
        return (x, y)

    def get_direction_vector(self, circumcenter, p_other, pa, pb, m):
        """
        根據斜率 m，計算方向向量 (dx, dy)
        """

        # 需要兩個相反的方向 u1, u2
        if m is None: # 垂直線
            u1 = (0, 1)
            u2 = (0, -1)
        else:
            # 一般斜率 m, 方向向量 (1, m)、(-1, -m)
            normoalization = (1**2 + m**2)**0.5
            u1 = (1 / normoalization, m / normoalization)
            u2 = (-1 / normoalization, -m / normoalization)

        if circumcenter is None:
            return u1

        def is_obtuse_at(P, A, B):
            # P, A, B 都是 點的 (x, y) 座標
            # PA . PB = |PA||PB|cosθ
            # 如果 cosθ < 0 => θ 是鈍角
            # 所以只需要計算內積是否小於 0
            v1 = (A[0] - P[0], A[1] - P[1])
            v2 = (B[0] - P[0], B[1] - P[1])
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            return dot < 0  # True = 以 P 為頂點的角是鈍角
        
        cx, cy = circumcenter  # 外心座標

        # 側是剛剛找的兩個方向的測試點
        test1 = (cx + u1[0], cy + u1[1])
        test2 = (cx + u2[0], cy + u2[1])

        # 計算兩個測試點到 p_other 的距離平方
        def d2(p, q):
            return (p[0]-q[0])**2 + (p[1]-q[1])**2

        # 只要pa pb其中一點是鈍角頂點，那麼外心就在外面，會需要往Delaunay triangulation的邊方向畫
        if (is_obtuse_at(pa, pb, p_other) or is_obtuse_at(pb, pa, p_other)):
            if d2(test1, pa) < d2(test2, pa):
                correct_u = u1
            else:
                correct_u = u2
        elif d2(test1, p_other) > d2(test2, p_other):
            correct_u = u1
        else:
            correct_u = u2

        return correct_u

    def get_ray_endpoint(self, circumcenter, p_other, pa, pb, m, b):
        """
        從 circumcenter 出發，沿著「遠離 p_other」的方向向量，
        取得這條射線和畫布矩形的「最遠邊界點」。

        如果這個方向射線完全不會碰到畫布，就退回用方向向量 * 2000。
        """
        cx, cy = circumcenter
        ux, uy = self.get_direction_vector(circumcenter, p_other, pa, pb, m)

        # 避免 direction vector 剛好是 0
        if ux == 0 and uy == 0:
            return circumcenter
        
        end_x = cx + ux * 100000
        end_y = cy + uy * 100000
        return (end_x, end_y)

        # xmin, xmax = 0, self.width+50
        # ymin, ymax = 0, self.height+50

        # candidates = []

        # # 與 x = xmin / xmax 的交點（只收 t > 0）
        # if ux != 0:
        #     # x = xmin
        #     t = (xmin - cx) / ux
        #     if t > 0:
        #         y = cy + t * uy
        #         if ymin <= y <= ymax:
        #             candidates.append((t, xmin, y))

        #     # x = xmax
        #     t = (xmax - cx) / ux
        #     if t > 0:
        #         y = cy + t * uy
        #         if ymin <= y <= ymax:
        #             candidates.append((t, xmax, y))

        # # 與 y = ymin / ymax 的交點（只收 t > 0）
        # if uy != 0:
        #     # y = ymin
        #     t = (ymin - cy) / uy
        #     if t > 0:
        #         x = cx + t * ux
        #         if xmin <= x <= xmax:
        #             candidates.append((t, x, ymin))

        #     # y = ymax
        #     t = (ymax - cy) / uy
        #     if t > 0:
        #         x = cx + t * ux
        #         if xmin <= x <= xmax:
        #             candidates.append((t, x, ymax))

        # # ✅ 如果這個方向完全不會撞到畫布，就用「乘 2000」的方式
        # if not candidates:
        #     end_x = cx + ux * 2000
        #     end_y = cy + uy * 2000
        #     return (end_x, end_y)

        # # ✅ 方向有穿過畫布：取「同一方向上最遠的那個」交點
        # t_max, ex, ey = max(candidates, key=lambda c: c[0])
        # return (ex, ey)
    
    def get_boundary_points(self, m, b):
        """
        計算直線 (y=mx+b 或 x=b) 與畫布矩形邊界的兩個交點。
        會檢查四個邊界：Left(0), Right(W), Top(0), Bottom(H)
        """
        W = self.width + 50
        H = self.height + 50
        points = []

        y_top = -100000
        y_bottom = 100000

        if m is None: # 垂直線 x = b
            return ((b, y_top), (b, y_bottom))
        elif abs(m) < 1e-6:
            return ((-100000, b), (100000, b))

        # y = m * x + b
        # m * x = y - b = > x = (y - b) / m
        x_top = (y_top - b) / m
        x_bottom = (y_bottom - b) / m

        return ((x_top, y_top), (x_bottom, y_bottom))

        # else: # 一般直線 y = mx + b
        #     # 檢查左牆 (x=0) -> y = b
        #     y_at_0 = b
        #     if 0 <= y_at_0 <= H:
        #         points.append((0, y_at_0))

        #     # 檢查右牆 (x=W) -> y = mW + b
        #     y_at_W = m * W + b
        #     if 0 <= y_at_W <= H:
        #         points.append((W, y_at_W))

        #     # 檢查上牆 (y=0) -> 0 = mx + b -> x = -b/m
        #     if abs(m) > 1e-9: # 避免除以零 (水平線)
        #         x_at_0 = -b / m
        #         if 0 < x_at_0 < W: # 使用 < 避免角落重複添加
        #             points.append((x_at_0, 0))

        #     # 檢查下牆 (y=H) -> H = mx + b -> x = (H-b)/m
        #     if abs(m) > 1e-9:
        #         x_at_H = (H - b) / m
        #         if 0 < x_at_H < W:
        #             points.append((x_at_H, H))
        
        # # 如果因為浮點誤差只找到不到2個點，退回到原始粗暴算法
        # if len(points) < 2:
        #     y1 = m * 0 + b
        #     y2 = m * W + b
        #     return ((0, y1), (W, y2))
            
        # # 回傳前兩個找到的點 (通常也只有兩個)
        # return (points[0], points[1])

    def voronoi_1_point(self, point_sorted_by_x):
        W = WingedEdge()
        i = self.point_index[point_sorted_by_x[0]]
        pid = W.add_polygon(site_index=i)
        return W
    
    def voronoi_2_points(self, points_sorted_by_x):
        W = WingedEdge()

        p1, p2 = points_sorted_by_x[0], points_sorted_by_x[1]

        # 透過 mapping 找到這兩個點在 self.points 裡的全域 index
        i1 = self.point_index[p1]
        i2 = self.point_index[p2]

        # 先假設 pid1 對應 p1, pid2 對應 p2
        pid1 = W.add_polygon(site_index=i1)
        pid2 = W.add_polygon(site_index=i2)

        # 求垂直平分線 (Bisector)
        slope, intercept = self.get_perpendicular_bisector(p1, p2)

        # 定義中垂線的兩個端點 v1, v2 (這構成了有向邊 Edge)
        # if slope is None:   # 垂直線
            # x = intercept
            # v1 = (x, 0)
            # v2 = (x, self.height + 50)
        # else:
        #     # 為了確保方向一致性，建議這裡固定 X 的方向，例如從左到右
        #     x1 = -50
        #     y1 = slope * x1 + intercept
        #     x2 = self.width + 50
        #     y2 = slope * x2 + intercept
        #     v1 = (x1, y1)
        #     v2 = (x2, y2)
        v1, v2 = self.get_boundary_points(slope, intercept)

        # 加頂點
        j1 = W.add_vertex(v1[0], v1[1], 0)
        j2 = W.add_vertex(v2[0], v2[1], 0)

        # 如果 p1 (對應 pid1) 真的在 v1->v2 的左邊
        if is_left(v1, v2, p1):
            left_p = pid1
            right_p = pid2
        else:
            left_p = pid2
            right_p = pid1
        # ==========================================

        k = len(W.start_vertex)
        k = W.add_edge(
            start_v=j1, end_v=j2,
            left_p=left_p, right_p=right_p,
            cw_pred=k, ccw_pred=k, cw_succ=k, ccw_succ=k
        )

        # Set 查找
        W.set_edge_around_vertex(j1, k)
        W.set_edge_around_vertex(j2, k)
        W.set_edge_around_polygon(pid1, k)
        W.set_edge_around_polygon(pid2, k)

        return W
    
    def voronoi_3_points(self, points_sorted_by_x):
        W = WingedEdge() 

        p1, p2, p3 = points_sorted_by_x[0], points_sorted_by_x[1], points_sorted_by_x[2]

        i1 = self.point_index[p1]
        i2 = self.point_index[p2]
        i3 = self.point_index[p3]

        # 建立 3 個多邊形 (對應 p1, p2, p3)
        pid1 = W.add_polygon(site_index=i1)
        pid2 = W.add_polygon(site_index=i2)
        pid3 = W.add_polygon(site_index=i3)

        # 計算三條邊的垂直平分線
        line12 = self.get_perpendicular_bisector(p1, p2)
        line23 = self.get_perpendicular_bisector(p2, p3)
        line13 = self.get_perpendicular_bisector(p1, p3)

        circumcenter = self.get_line_intersection(line12, line23)  # 外心

        if circumcenter is None:
            # --- 三點共線 ---            
            m12, b12 = line12
            m23, b23 = line23
            
            # 處理 k12 (分隔 pid1 和 pid2)
            (x1_12, y1_12), (x2_12, y2_12) = self.get_boundary_points(m12, b12)
            v1_k12 = W.add_vertex(x1_12, y1_12, 0)
            v2_k12 = W.add_vertex(x2_12, y2_12, 0)

            # 使用 is_left 判斷 p1 和 p2 的相對位置
            if is_left((x1_12, y1_12), (x2_12, y2_12), p1):
                left_p = pid1
                right_p = pid2
            else:
                left_p = pid2
                right_p = pid1
            
            # 取得邊 ID
            k12 = len(W.start_vertex)
            W.add_edge(
                start_v=v1_k12, end_v=v2_k12,
                left_p=left_p, right_p=right_p,
                cw_pred=k12, ccw_pred=k12,  # 頂點上的自我迴圈
                cw_succ=k12, ccw_succ=k12   # 頂點上的自我迴圈
            )
            W.set_edge_around_vertex(v1_k12, k12)
            W.set_edge_around_vertex(v2_k12, k12)

            # 處理 k23 (分隔 pid2 和 pid3)
            (x1_23, y1_23), (x2_23, y2_23) = self.get_boundary_points(m23, b23)
            v1_k23 = W.add_vertex(x1_23, y1_23, 0)
            v2_k23 = W.add_vertex(x2_23, y2_23, 0)

            # 使用 is_left 判斷 p2 和 p3 的相對位置
            if is_left((x1_23, y1_23), (x2_23, y2_23), p2):
                left_p = pid2
                right_p = pid3
            else:
                left_p = pid3
                right_p = pid2

            # 取得邊 ID
            k23 = len(W.start_vertex)
            W.add_edge(
                start_v=v1_k23, end_v=v2_k23,
                left_p=left_p, right_p=right_p,  # 為了連貫, p2 必須在左
                cw_pred=k23, ccw_pred=k23,
                cw_succ=k23, ccw_succ=k23
            )
            W.set_edge_around_vertex(v1_k23, k23)
            W.set_edge_around_vertex(v2_k23, k23)

            # 設置多邊形查找
            W.set_edge_around_polygon(pid1, k12)
            W.set_edge_around_polygon(pid2, k12) # pid2 可由 k12 或 k23 開始
            W.set_edge_around_polygon(pid3, k23)
        else:
            # --- 三點不共線 (有外心) ---
            cx, cy = circumcenter

            # 將點 p1, p2, p3 和它們對應的 polygon id 放在一起
            # 並根據相對於外心的角度進行 CCW 排序
            
            # (假設 pid1=0, pid2=1, pid3=2)
            points_data = [
                {'p': p1, 'pid': pid1},
                {'p': p2, 'pid': pid2},
                {'p': p3, 'pid': pid3}
            ]

            points_with_angles = []
            for data in points_data:
                # 計算從外心指向點的向量
                vec_x = data['p'][0] - cx
                vec_y = data['p'][1] - cy
                
                # 使用 atan2 計算角度
                angle = math.atan2(vec_y, vec_x)
                points_with_angles.append((angle, data))

            # 會從第三象限開始到第二象限 (CCW)
            # 在左上(0, 0)的畫布中會是左上到右上再到右下再到左下
            points_with_angles.sort()
            sorted_data = [data for angle, data in points_with_angles]

            # 現在有了一個保證 CCW 順序的列表
            pid_a = sorted_data[0]['pid']
            pid_b = sorted_data[1]['pid']
            pid_c = sorted_data[2]['pid']
            
            p_a = sorted_data[0]['p']
            p_b = sorted_data[1]['p']
            p_c = sorted_data[2]['p']

            # 重新計算平分線，但這次是使用 CCW 順序
            line_ab = self.get_perpendicular_bisector(p_a, p_b)
            line_bc = self.get_perpendicular_bisector(p_b, p_c)
            line_ca = self.get_perpendicular_bisector(p_c, p_a)

            # 計算三條射線的終點
            (x_ab, y_ab) = self.get_ray_endpoint(circumcenter, p_c, p_a, p_b, line_ab[0], line_ab[1])
            (x_bc, y_bc) = self.get_ray_endpoint(circumcenter, p_a, p_b, p_c, line_bc[0], line_bc[1])
            (x_ca, y_ca) = self.get_ray_endpoint(circumcenter, p_b, p_c, p_a, line_ca[0], line_ca[1])

            # 建立頂點 (v_center 和射線終點)
            v_center = W.add_vertex(cx, cy)
            v_ray_ab = W.add_vertex(x_ab, y_ab, 0)
            v_ray_bc = W.add_vertex(x_bc, y_bc, 0)
            v_ray_ca = W.add_vertex(x_ca, y_ca, 0)

            # 預先規劃邊的 ID
            k_ab = len(W.start_vertex)
            k_bc = k_ab + 1
            k_ca = k_ab + 2

            # ---- 先決定每條邊的 left/right polygon ----
            # k_ab
            if is_left((cx, cy), (x_ab, y_ab), p_a):
                left_ab, right_ab = pid_a, pid_b
            else:
                left_ab, right_ab = pid_b, pid_a

            # k_bc
            if is_left((cx, cy), (x_bc, y_bc), p_b):
                left_bc, right_bc = pid_b, pid_c
            else:
                left_bc, right_bc = pid_c, pid_b

            # k_ca
            if is_left((cx, cy), (x_ca, y_ca), p_c):
                left_ca, right_ca = pid_c, pid_a
            else:
                left_ca, right_ca = pid_a, pid_c

            # 邊 k_ab (分隔 pid_a, pid_b)
            # 雖然畫布上下是反的，但這邊仍然依照數學上的 CCW 順序來設定前驅和後繼，這樣不用費心轉換
            W.add_edge(
                start_v=v_center, end_v=v_ray_ab,
                left_p=left_ab, right_p=right_ab,
                cw_pred=k_ca, ccw_pred=k_bc,  # CCW 順序: ab -> bc -> ca
                cw_succ=k_ab, ccw_succ=k_ab   # 射線終點自我迴圈
            )

            # 邊 k_bc (分隔 pid_b, pid_c)
            W.add_edge(
                start_v=v_center, end_v=v_ray_bc,
                left_p=left_bc, right_p=right_bc,
                cw_pred=k_ab, ccw_pred=k_ca,  # CCW 順序: ab -> bc -> ca
                cw_succ=k_bc, ccw_succ=k_bc
            )

            # 邊 k_ca (分隔 pid_c, pid_a)
            W.add_edge(
                start_v=v_center, end_v=v_ray_ca,
                left_p=left_ca, right_p=right_ca,
                cw_pred=k_bc, ccw_pred=k_ab,  # CCW 順序: ab -> bc -> ca
                cw_succ=k_ca, ccw_succ=k_ca
            )

            # 設置查找連結
            W.set_edge_around_vertex(v_center, k_ab)
            W.set_edge_around_vertex(v_ray_ab, k_ab)
            W.set_edge_around_vertex(v_ray_bc, k_bc)
            W.set_edge_around_vertex(v_ray_ca, k_ca)

            W.set_edge_around_polygon(pid_a, k_ab) # pid_a 與 k_ab 和 k_ca 相鄰
            W.set_edge_around_polygon(pid_b, k_bc) # pid_b 與 k_bc 和 k_ab 相鄰
            W.set_edge_around_polygon(pid_c, k_ca) # pid_c 與 k_ca 和 k_bc 相鄰

        return W
    
    def save_snapshot(self, step_type, vd_obj, extra_info=None, deepcopyObj=True):
        snapshot = {
            'type': step_type,
            'vd': copy.deepcopy(vd_obj) if deepcopyObj else vd_obj, # 複製整個拓樸結構
            'info': extra_info
        }
        self.history.append(snapshot)

    def build_voronoi_dc(self, points_sorted_by_x) -> tuple[WingedEdge, list]:    
        n = len(points_sorted_by_x)
        
        # current_ponts = points_sorted_by_x
        if n == 1:
            current_result = self.voronoi_1_point(points_sorted_by_x)
            current_hull = self.convex_hull_from_voronoi(current_result, self.points)
        elif n == 2:
            current_result = self.voronoi_2_points(points_sorted_by_x)
            current_hull = self.convex_hull_from_voronoi(current_result, self.points)
        elif n == 3:
            current_result = self.voronoi_3_points(points_sorted_by_x)
            current_hull = self.convex_hull_from_voronoi(current_result, self.points)
        else:
            mid = n // 2 + n % 2
            
            # 接收 (WingedEdge, points_list) 元組
            (left_VD, left_H) = self.build_voronoi_dc(points_sorted_by_x[:mid])
            (right_VD, right_H) = self.build_voronoi_dc(points_sorted_by_x[mid:])

            # 將兩個圖和兩個點集傳入 merge 函數
            (merged_VD, merged_H) = self.merge_voronoi(left_VD, left_H, right_VD, right_H)
            current_result = merged_VD
            current_hull = merged_H
        
        return (current_result, current_hull)

    def convex_hull_from_voronoi(self, we: WingedEdge, all_points):
        # 取得亂序的索引
        site_indices = self.hull_site_indices_from_voronoi(we)
        
        # 包裝成 (x, y, i) -> 這裡 index 依然是必要的！這樣只要傳回索引的列表就好
        pts = [(all_points[i][0], all_points[i][1], i) for i in site_indices]
        
        if len(pts) < 3:
            return [p[2] for p in pts] # 直接回傳索引

        # 計算重心 (Centroid)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        # 只需依照角度排序
        pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

        # 取出索引回傳
        return [p[2] for p in pts]

    def draw_voronoi_wingededge(self, we: WingedEdge, offset_x=0, offset_y=0, color='black', tag="voronoi_edge"):
        m = len(we.start_vertex)
        for k in range(m):
            v1_idx = we.start_vertex[k]
            v2_idx = we.end_vertex[k]

            if v1_idx == None or v2_idx == None:
                continue
            
            if we.x_vertex[v1_idx] is None or we.y_vertex[v1_idx] is None or we.x_vertex[v2_idx] is None or we.y_vertex[v2_idx] is None:
                continue
            x1 = we.x_vertex[v1_idx] + offset_x
            y1 = we.y_vertex[v1_idx] + offset_y
            x2 = we.x_vertex[v2_idx] + offset_x
            y2 = we.y_vertex[v2_idx] + offset_y

            self.canvas.create_line(x1, y1, x2, y2, fill=color, tags=tag)

            clipped = self.clip_segment_to_canvas(x1, y1, x2, y2, self.width, self.height)
            if clipped is None: continue

            cx1, cy1, cx2, cy2 = clipped
            mid_x = (cx1 + cx2) / 2
            mid_y = (cy1 + cy2) / 2

            self.canvas.create_text(
                mid_x, mid_y,
                text=str(k),
                fill="red",
                font=("Arial", 10),
                tags=tag
            )

        for i in range(len(we.edge_around_polygon)):
            idx = we.site_of_polygon[i]
            x, y = self.points[idx]

            self.canvas.create_text(
                x+10, y,
                text=str(i),
                fill="blue",
                font=("Arial", 10),
                tags=tag
            )

        for j in range(len(we.x_vertex)):
            if j is None: continue
            x = we.x_vertex[j]
            y = we.y_vertex[j]

            if x is None or y is None: continue

            self.canvas.create_text(
                x, y,
                text=str(j),
                fill="purple",
                font=("Arial", 15),
                tags=tag
            )

        self.root.update_idletasks()
        self.root.update()

    def hull_site_indices_from_voronoi(self, we: WingedEdge) -> list[int]:
        hull_sites = set()

        m = len(we.start_vertex)
        for k in range(m):
            if k is None: continue
            v1 = we.start_vertex[k]
            v2 = we.end_vertex[k]

            if v1 is None or v2 is None or we.w_vertex[v1] is None or we.w_vertex[v2] is None:
                continue

            # edge 有無限遠點
            if we.w_vertex[v1] == 0 or we.w_vertex[v2] == 0:
                for poly in (we.left_polygon[k], we.right_polygon[k]):
                    if poly is None:
                        continue
                    site_idx = we.site_of_polygon[poly]
                    if site_idx is not None:
                        hull_sites.add(site_idx)

        return list(hull_sites)

    def merge_voronoi(self, left_VD: WingedEdge, left_hull: list, right_VD: WingedEdge, right_hull: list) -> tuple[WingedEdge, list]:
        # 最終合併的圖
        VD_merged = WingedEdge()
        
        VD_merged, v_off, p_off, e_off = self.merge_structures(left_VD, right_VD)

        left_edge_start  = 0
        left_edge_end    = len(left_VD.start_vertex)              # 左邊 edge ID: [0, left_edge_end)

        right_edge_start = e_off
        right_edge_end   = e_off + len(right_VD.start_vertex)      # 右邊 edge ID: [right_edge_start, right_edge_end)

        if DEBUG:
            print("==================================================================")

        # 順序 ['left_hull', 'right_hull', 'left_voronoi', right_voronoi', 'merged_hull', 'seam_edges', 'remove_line', 'merged_voronoi']
        # merged_hull 和 remove_line 搭配沒有消線的 merged_voronoi
        self.save_snapshot(
            step_type='left_hull',
            vd_obj=left_VD,
            extra_info=left_hull,
            deepcopyObj=False
        )
        self.save_snapshot(
            step_type='right_hull',
            vd_obj=right_VD,
            extra_info=right_hull,
            deepcopyObj=False
        )
        self.save_snapshot(
            step_type='left_voronoi',
            vd_obj=left_VD,
            extra_info=None,
            deepcopyObj=False
        )
        self.save_snapshot(
            step_type='right_voronoi',
            vd_obj=right_VD,
            extra_info=None,
            deepcopyObj=False
        )
                
        iu, ju = self.upper_tangent(left_hull, right_hull)
        il, jl = self.lower_tangent(left_hull, right_hull)

        merged_hull = []
        curr = il
        while True:
            merged_hull.append(left_hull[curr])
            if curr == iu:
                break
            curr = (curr + 1) % len(left_hull) # 逆時針移動

        # 右凸包部分：從 上切點(ju) 順著逆時針走到 下切點(jl)
        curr = ju
        while True:
            merged_hull.append(right_hull[curr])
            if curr == jl:
                break
            curr = (curr + 1) % len(right_hull) # 逆時針移動

        self.save_snapshot(
            step_type='merged_hull',
            vd_obj=None,
            extra_info=merged_hull,
            deepcopyObj=False
        )

        # 找出起點 Site Index
        l_site_idx = left_hull[iu]
        r_site_idx = right_hull[ju]
        
        # 找出終點 Site Index (用於終止條件)
        l_bottom_idx = left_hull[il]
        r_bottom_idx = right_hull[jl]

        if DEBUG:
            # left_hull 和 right_hull 裡的點是全域點的 point
            left_hull_points = [self.points[idx] for idx in left_hull]
            right_hull_points = [self.points[idx] for idx in right_hull]
            print(f"Left Hull Points: {left_hull_points}")
            print(f"Right Hull Points: {right_hull_points}")

        site_to_poly_map = {}
        for pid, site_idx in enumerate(VD_merged.site_of_polygon):
            if site_idx is not None:
                site_to_poly_map[site_idx] = pid

        l_poly_id = site_to_poly_map.get(l_site_idx)
        r_poly_id = site_to_poly_map.get(r_site_idx)

        # 初始化
        last_v_id = None
        last_edge_id = None
        last_chosen_side = None
        last_hit_edge = None
        last_hit_edge_extra = None
        seam_edges = []

        do_remove_lines = True

        # 進入縫合迴圈
        while True:
            # debug將左右的voronoi都畫出來(先清除畫布)
            if DEBUG:
                self.canvas.delete("voronoi_edge")
                self.draw_voronoi_wingededge(VD_merged, offset_x=0, offset_y=0, color='green')
                # print(VD_merged.edges_dataframe())

            p_l = self.points[l_site_idx]
            p_r = self.points[r_site_idx]

            if DEBUG:
                print(f"Upper Tangent Points: Left Site Index {self.points[l_site_idx]}, Right Site Index {self.points[r_site_idx]}")
                print(f"Lower Tangent Points: Left Site Index {self.points[l_bottom_idx]}, Right Site Index {self.points[r_bottom_idx]}")
            
            bisector = self.get_perpendicular_bisector(p_l, p_r)  # (slope, intercept)

            if last_v_id is None:
                m, b = bisector
                virtual_start_y = -100000.0

                if m is None: 
                    last_x = b
                    last_y = virtual_start_y
                    
                elif abs(m) < 1e-9:
                    last_x = -100000.0
                    last_y = b
                    
                else:
                    # 【一般斜線】 y = mx + b  =>  x = (y - b) / m
                    # 算出這條線在 y = -100000 時的 x 是多少
                    last_x = (virtual_start_y - b) / m
                    last_y = virtual_start_y
                
                last_v_id = VD_merged.add_vertex(last_x, last_y, 0)  # 新增無限遠的點
            else:
                last_x = VD_merged.x_vertex[last_v_id]
                last_y = VD_merged.y_vertex[last_v_id]

            candidate_l = self.get_first_collision(VD_merged, l_poly_id, bisector, last_y=last_y, last_x=last_x, edge_range=(left_edge_start, left_edge_end))
            candidate_r = self.get_first_collision(VD_merged, r_poly_id, bisector, last_y=last_y, last_x=last_x, edge_range=(right_edge_start, right_edge_end))
            
            chosen_side = None
            collision_info = None # (point, edge_id)
            collision_info_both = None

            # 兩邊都沒撞到 -> 這是最後一條通往無限遠的縫合線
            if candidate_l is None and candidate_r is None:  # 這邊不加入seam_edges，因為等於左右都沒有連接
                # 加上最後一條無限長邊後跳出
                seam_edge = self.add_final_ray(VD=VD_merged,
                                   start_v=last_v_id,
                                   bisector=bisector, 
                                   l_poly=l_poly_id,
                                   r_poly=r_poly_id,
                                   last_hit_edge_id=last_hit_edge,
                                   last_hit_edge_id_extra=last_hit_edge_extra,
                                   new_seam_id=None,
                                   last_seam_id=last_edge_id,
                                   last_chosen_side=last_chosen_side,
                                   last_hit_point=last_v_id,
                                   last_hit_point_extra=None)
                seam_edges.append(seam_edge)
                do_remove_lines = False
                break

            dist_l = candidate_l[0] if candidate_l else float('inf')
            dist_r = candidate_r[0] if candidate_r else float('inf')

            if abs(dist_l - dist_r) < 1e-9:
                chosen_side = 'BOTH'
                collision_info_both = (candidate_l[1:], candidate_r[1:]) # ((point, edge_id), (point, edge_id))
                collision_info = collision_info_both[0]  # 任選一個交點 (理論上應該相同)
            elif dist_l < dist_r:
                chosen_side = 'LEFT'
                collision_info = candidate_l[1:] # (point, edge_id)
            else:
                chosen_side = 'RIGHT'
                collision_info = candidate_r[1:]

            intersect_pt, hit_edge_k = collision_info


            # 建立新的 Voronoi 頂點 (位於交點處)
            new_v_id = VD_merged.add_vertex(intersect_pt[0], intersect_pt[1])

            if is_left((VD_merged.x_vertex[last_v_id], VD_merged.y_vertex[last_v_id]),
                       (VD_merged.x_vertex[new_v_id], VD_merged.y_vertex[new_v_id]),
                       self.points[VD_merged.site_of_polygon[l_poly_id]]):
                left_pid = l_poly_id
                right_pid = r_poly_id
            else:
                left_pid = r_poly_id
                right_pid = l_poly_id

            # 這條邊分隔了目前的 l_poly 和 r_poly
            k = len(VD_merged.start_vertex)
            new_edge_id = VD_merged.add_edge(
                start_v=last_v_id, end_v=new_v_id,
                left_p=left_pid, right_p=right_pid,
                cw_pred=k, ccw_pred=k, cw_succ=k, ccw_succ=k
            )

            seam_edges.append(new_edge_id)

            # 更新頂點的 edge_around
            if VD_merged.edge_around_vertex[last_v_id] is None:
                VD_merged.set_edge_around_vertex(last_v_id, new_edge_id)
            if VD_merged.edge_around_vertex[new_v_id] is None:
                VD_merged.set_edge_around_vertex(new_v_id, new_edge_id)

            if DEBUG:
                print("hit edge:", hit_edge_k)

            if last_edge_id is not None:
                self.update_edge_topology(VD=VD_merged,
                                          new_seam_id=new_edge_id,
                                          last_seam_id=last_edge_id,
                                          last_chosen_side=last_chosen_side,
                                          last_hit_edge_k=last_hit_edge,
                                          last_hit_edge_k_extra=last_hit_edge_extra,
                                          last_hit_point=last_v_id,
                                          last_hit_point_extra=last_hit_point_extra)

            last_v_id = new_v_id
            last_edge_id = new_edge_id
            last_chosen_side = chosen_side
            last_hit_edge = hit_edge_k
            last_hit_edge_extra = collision_info_both[1][1] if collision_info_both else None
            last_hit_point = last_v_id
            last_hit_point_extra = collision_info_both[1][0] if collision_info_both else None

            # 剪斷舊邊並切換多邊形
            if chosen_side == 'BOTH':
                hit_edge_k_l = collision_info_both[0][1]
                hit_edge_k_r = collision_info_both[1][1]

                # 左邊多邊形的下一個多邊形
                next_l_poly_id = self.get_neighbor_across_edge(VD_merged, hit_edge_k_l, l_poly_id)
                # 右邊多邊形的下一個多邊形
                next_r_poly_id = self.get_neighbor_across_edge(VD_merged, hit_edge_k_r, r_poly_id)
                
                # 修剪邊 (先左後右)
                self.clip_edge(VD_merged, hit_edge_k_r, new_v_id, 'RIGHT', self.points[l_site_idx], self.points[r_site_idx])
                self.clip_edge(VD_merged, hit_edge_k_l, new_v_id, 'LEFT', self.points[l_site_idx], self.points[r_site_idx])

                # 切換狀態
                l_poly_id = next_l_poly_id
                r_poly_id = next_r_poly_id
                l_site_idx = VD_merged.site_of_polygon[l_poly_id]
                r_site_idx = VD_merged.site_of_polygon[r_poly_id]
            elif chosen_side == 'LEFT':
                # 找出下一個多邊形 (跨過 edge 的那個)
                next_poly_id = self.get_neighbor_across_edge(VD_merged, hit_edge_k, l_poly_id)
                
                # 修剪邊
                self.clip_edge(VD_merged, hit_edge_k, new_v_id, 'LEFT', self.points[l_site_idx], self.points[r_site_idx])
                # VD_merged.ccw_successor[new_edge_id] = hit_edge_k

                # 切換狀態
                l_poly_id = next_poly_id
                l_site_idx = VD_merged.site_of_polygon[l_poly_id]
            else: # RIGHT
                next_poly_id = self.get_neighbor_across_edge(VD_merged, hit_edge_k, r_poly_id)

                self.clip_edge(VD_merged, hit_edge_k, new_v_id, 'RIGHT', self.points[l_site_idx], self.points[r_site_idx])
                                
                r_poly_id = next_poly_id
                r_site_idx = VD_merged.site_of_polygon[r_poly_id]

            if DEBUG:
                self.canvas.delete("voronoi_edge")
                self.draw_voronoi_wingededge(VD_merged, offset_x=0, offset_y=0, color='green')
                # print(VD_merged.edges_dataframe())

            # 檢查是否到達下公切線 (終止條件)
            if l_site_idx == l_bottom_idx and r_site_idx == r_bottom_idx:
                final_bisector = self.get_perpendicular_bisector(self.points[l_site_idx], self.points[r_site_idx])
                
                seam_edge = self.add_final_ray(VD=VD_merged,
                                   start_v=last_v_id,
                                   bisector=final_bisector, 
                                   l_poly=l_poly_id,
                                   r_poly=r_poly_id,
                                   last_hit_edge_id=last_hit_edge,
                                   last_hit_edge_id_extra=last_hit_edge_extra,
                                   new_seam_id=new_edge_id,
                                   last_seam_id=last_edge_id,
                                   last_chosen_side=last_chosen_side,
                                   last_hit_point=last_v_id,
                                   last_hit_point_extra=last_hit_point_extra)
                seam_edges.append(seam_edge)
                break

        # 順序 ['left_hull', 'right_hull', 'left_voronoi', right_voronoi', 'merged_hull', 'seam_edges', 'remove_line', 'merged_voronoi']
        # merged_hull 和 remove_line 搭配沒有消線的 merged_voronoi
        self.save_snapshot(
            step_type='seam_edges',
            vd_obj=VD_merged,
            extra_info=seam_edges,
            deepcopyObj=True
        )
        if do_remove_lines:
            self.remove_disconnected_components(VD=VD_merged, seam_edges=seam_edges)
        self.save_snapshot(
            step_type='remove_line',
            vd_obj=VD_merged,
            extra_info=None,
            deepcopyObj=False
        )
        self.save_snapshot(
            step_type='merged_voronoi',
            vd_obj=VD_merged,
            extra_info=None,
            deepcopyObj=False
        )

        return VD_merged, merged_hull

    def add_final_ray(self, VD, start_v, bisector, l_poly, r_poly, last_hit_edge_id, last_hit_edge_id_extra, new_seam_id, last_seam_id, last_chosen_side, last_hit_point, last_hit_point_extra):
        # 計算無限遠的終點
        m, b = bisector
        virtual_end_y = 100000.0  # 向下無限遠

        if m is None:
            end_x = b
            end_y = virtual_end_y
        elif abs(m) < 1e-9:
            end_y = b
            if last_seam_id is None:
                end_x = 100000.0
            else:
                last_seam_s = VD.start_vertex[last_seam_id]
                last_seam_e = VD.end_vertex[last_seam_id]
                
                last_start_x = VD.x_vertex[last_seam_s]
                last_end_x = VD.x_vertex[last_seam_e]
                
                direction_x = self.points[VD.site_of_polygon[l_poly]][0]

                eps = 1e-9
                dx = last_end_x - direction_x
                if dx > eps:
                    end_x = -100000.0
                elif dx < eps:
                    end_x = 100000.0
                else:
                    if last_end_x < last_start_x:
                        end_x = -100000.0
                    else:
                        end_x = 100000.0
        else:
            end_x = (virtual_end_y - b) / m
            end_y = virtual_end_y
            
        end_v = VD.add_vertex(end_x, end_y, 0) # w=0 代表虛擬無限點

        start_p = (VD.x_vertex[start_v], VD.y_vertex[start_v])
        r_poly_p = self.points[VD.site_of_polygon[r_poly]]

        if is_left(start_p, (end_x, end_y), r_poly_p):
            l_poly, r_poly = r_poly, l_poly

        # 建立最後一條縫合線
        final_edge_id = VD.add_edge(
            start_v=start_v, end_v=end_v,
            left_p=l_poly, right_p=r_poly,
            cw_pred=-1, ccw_pred=-1, cw_succ=-1, ccw_succ=-1
        )
        
        # 設定頂點查找表
        if VD.edge_around_vertex[start_v] is None:
            VD.set_edge_around_vertex(start_v, final_edge_id)
        if VD.edge_around_vertex[end_v] is None:
            VD.set_edge_around_vertex(end_v, final_edge_id)

        VD.cw_successor[final_edge_id] = final_edge_id
        VD.ccw_successor[final_edge_id] = final_edge_id

        if last_seam_id is not None:
                self.update_edge_topology(VD=VD,
                                          new_seam_id=final_edge_id,
                                          last_seam_id=last_seam_id,
                                          last_chosen_side=last_chosen_side,
                                          last_hit_edge_k=last_hit_edge_id,
                                          last_hit_edge_k_extra=last_hit_edge_id_extra,
                                          last_hit_point=last_hit_point,
                                          last_hit_point_extra=last_hit_point_extra)

        if VD.left_polygon[final_edge_id] is not None:
            VD.edge_around_polygon[VD.left_polygon[final_edge_id]] = final_edge_id
        if VD.right_polygon[final_edge_id] is not None:
            VD.edge_around_polygon[VD.right_polygon[final_edge_id]] = final_edge_id

        return final_edge_id

    def update_edge_topology(self, VD: WingedEdge, 
                             new_seam_id: int, 
                             last_seam_id: int, 
                             last_chosen_side: str, 
                             last_hit_edge_k: int,
                             last_hit_edge_k_extra: int, # 用於 BOTH 情況的第二條邊
                             last_hit_point: int,
                             last_hit_point_extra: int,
                             new_v_id: int = None,
                             l_poly_id: int = None,
                             r_poly_id: int = None):
        
        def is_last_hit_start(VD: WingedEdge, last_hit_edge_k, last_hit_point):
            return VD.start_vertex[last_hit_edge_k] == last_hit_point


        # 沒撞的那邊是連續的，撞到的那邊是斷裂的。
        if last_seam_id is not None:
            if last_chosen_side == 'LEFT':
                # current seam connection
                VD.cw_predecessor[new_seam_id] = last_seam_id
                VD.ccw_predecessor[new_seam_id] = last_hit_edge_k

                # last seam connection
                VD.cw_successor[last_seam_id] = last_hit_edge_k
                VD.ccw_successor[last_seam_id] = new_seam_id

                # last hit edge
                if is_last_hit_start(VD, last_hit_edge_k, last_hit_point):
                    VD.ccw_predecessor[last_hit_edge_k] = last_seam_id
                    VD.cw_predecessor[last_hit_edge_k] = new_seam_id
                else:
                    VD.ccw_successor[last_hit_edge_k] = last_seam_id
                    VD.cw_successor[last_hit_edge_k] = new_seam_id

            elif last_chosen_side == 'RIGHT':
                # current seam connection
                VD.cw_predecessor[new_seam_id] = last_hit_edge_k
                VD.ccw_predecessor[new_seam_id] = last_seam_id

                # last seam connection
                VD.cw_successor[last_seam_id] = new_seam_id
                VD.ccw_successor[last_seam_id] = last_hit_edge_k

                # last hit edge
                if is_last_hit_start(VD, last_hit_edge_k, last_hit_point):
                    VD.ccw_predecessor[last_hit_edge_k] = new_seam_id
                    VD.cw_predecessor[last_hit_edge_k] = last_seam_id
                else:
                    VD.ccw_successor[last_hit_edge_k] = new_seam_id
                    VD.cw_successor[last_hit_edge_k] = last_seam_id

            elif last_chosen_side == 'BOTH':
                # current seam connection
                VD.cw_predecessor[new_seam_id] = last_hit_edge_k_extra
                VD.ccw_predecessor[new_seam_id] = last_hit_edge_k

                # last seam connection
                VD.cw_successor[last_seam_id] = last_hit_edge_k
                VD.ccw_successor[last_seam_id] = last_hit_edge_k_extra

                # last hit edge
                if is_last_hit_start(VD, last_hit_edge_k, last_hit_point):
                    VD.ccw_predecessor[last_hit_edge_k] = last_seam_id
                    VD.cw_predecessor[last_hit_edge_k] = new_seam_id
                else:
                    VD.ccw_successor[last_hit_edge_k] = last_seam_id
                    VD.cw_successor[last_hit_edge_k] = new_seam_id

                # last hit edge extra
                if is_last_hit_start(VD, last_hit_edge_k_extra, last_hit_point_extra):
                    VD.ccw_predecessor[last_hit_edge_k_extra] = new_seam_id
                    VD.cw_predecessor[last_hit_edge_k_extra] = last_seam_id
                else:
                    VD.ccw_successor[last_hit_edge_k_extra] = new_seam_id
                    VD.cw_successor[last_hit_edge_k_extra] = last_seam_id

    def get_neighbor_across_edge(self, vd, edge_k, current_poly_id):
        if vd.left_polygon[edge_k] == current_poly_id:
            return vd.right_polygon[edge_k]
        return vd.left_polygon[edge_k]

    def get_line_from_points(self, p1: tuple, p2: tuple):
        """
        根據兩點計算直線方程式
        回傳 (slope, intercept)
        若為垂直線，回傳 (None, x_intercept)
        """
        x1, y1 = p1
        x2, y2 = p2

        if abs(x2 - x1) < 1e-9:
            return (None, x1)  # 垂直線
        else:
            slope = (y2 - y1) / (x2 - x1)
            # y = slope * x + intercept  =>  intercept = y - slope * x
            intercept = y1 - slope * x1
            return (slope, intercept)

    def get_line_intersection_with_2point(self, line: tuple, segment: tuple):
        """
        Args:
            line: (slope, intercept) Bisector
            segment: ((x1, y1), (x2, y2)) 
        
        Returns:
            (x, y) 如果交點在有效範圍內
            None   如果交點在範圍外
        """
        m, b = line
        (x1, y1), (x2, y2) = segment

        # 先解出數學上的交點 (無限直線的交點)
        x_int, y_int = None, None

        if m is None: # Bisector 垂直
            x_int = b
            if abs(x2 - x1) < 1e-9: return None # 兩條都垂直
            slope_seg = (y2 - y1) / (x2 - x1)
            y_int = slope_seg * x_int + (y1 - slope_seg * x1)
        else: # Bisector 一般線
            if abs(x2 - x1) < 1e-9: # Segment 垂直
                x_int = x1
                y_int = m * x1 + b
            else:
                slope_seg = (y2 - y1) / (x2 - x1)
                if abs(m - slope_seg) < 1e-9: return None # 平行
                # y1 = slope_seg * x1 + intercept_seg
                # intercept_seg = y1 - slope_seg * x1
                # 解聯立方程式
                # m * x_int + b = slope_seg * x_int + (y1 - slope_seg * x1)
                # (m - slope_seg) * x_int = (y1 - slope_seg * x1) - b
                intercept_seg = y1 - slope_seg * x1
                x_int = (intercept_seg - b) / (m - slope_seg)
                y_int = m * x_int + b

        return (x_int, y_int)

    def get_first_collision(self, VD: WingedEdge, poly_id: int, bisector: tuple, last_y: float, last_x: float, edge_range: tuple):
        """
        Args:
            VD: WingedEdge 物件
            poly_id: 目標多邊形 ID
            bisector: (slope, intercept) 
            last_y: 上一個縫合點的 Y 座標 (用來過濾掉往回跑的交點)
        
        Returns:
            (distance, (x, y), edge_k)
        """
        min_dist_sq = float('inf')
        collision_pt = None
        collision_edge = None

        # 1. 找出起始邊
        # start_edge_k = VD.edge_around_polygon[poly_id]
        # if start_edge_k is None: return None # 孤立點或錯誤
        
        # curr_edge_k = start_edge_k

        for curr_edge_k in range(edge_range[0], edge_range[1]):
            if VD.left_polygon[curr_edge_k] != poly_id and VD.right_polygon[curr_edge_k] != poly_id:
                continue
            seg_p1, seg_p2 = self.get_clipped_edge_segment(VD, curr_edge_k)  # 多邊形的第一條邊(本來就是裁切後)

            # if (seg_p1 is None and seg_p2 is None) or seg_p1 == (None, None, None) or seg_p2 == (None, None, None):
            #     continue

            if seg_p1 is None and seg_p2 is None:
                continue

            intersect = self.get_line_intersection_with_2point(bisector, (seg_p1[:2], seg_p2[:2]))  # 回傳 (x, y) 或 None

            if intersect:  # 有交點(沒有會回 None)
                ix, iy = intersect
                
                # 交點必須在「當前縫合高度」之下 (因為縫合線是從上往下掃描，上面是0下面是正無限)
                if iy >= last_y - 1e-9:
                    
                    # 這邊已經是確定無限延長的線會有交點了，所以只要確認交點在線段上
                    if self.is_point_on_segment((intersect[0], intersect[1], 1), seg_p1, seg_p2):
                        # dist_metric = iy
                        # 計算與上一點的距離平方
                        dx = ix - last_x
                        dy = iy - last_y
                        dist_sq = dx*dx + dy*dy
                        
                        # 防止選到 "上一點" 自己
                        # 如果距離太近 (例如 < 1e-9)，視為原地踏步，跳過
                        if dist_sq > 1e-9 and dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            collision_pt = intersect
                            collision_edge = curr_edge_k

        if collision_pt:
            return (min_dist_sq, collision_pt, collision_edge)
        return None

    def is_point_on_segment(self, p, a, b):
        """
        判斷點 p 是否在線段 a-b 上（只做範圍檢查，不含共線性）。
        p, a, b 都是 (x, y, w) tuple。
        其中：
            w = 1  -> 普通有限點
            w = 0   -> 無限遠點
        規則：
            - 無限遠點視為在線段上；
            - 範圍檢查時，只有「有限端點」會形成邊界，
            如果端點是無限遠點，對應那一側就不設邊界（代表向那邊無限延伸）。
        """
        px, py, wp = p
        ax, ay, wa = a
        bx, by, wb = b

        epsilon = 1e-9

        # X 方向的範圍檢查（依照左右端點是否有限決定是否做邊界）
        if ax <= bx:
            left_x, left_w = ax, wa
            right_x, right_w = bx, wb
        else:
            left_x, left_w = bx, wb
            right_x, right_w = ax, wa

        # 左邊端點是有限點 => 檢查 px 不可小於它
        if left_w != 0:
            if px < left_x - epsilon:
                return False

        # 右邊端點是有限點 => 檢查 px 不可大於它
        if right_w != 0:
            if px > right_x + epsilon:
                return False

        # Y 方向的範圍檢查（同樣只對有限端點做邊界）
        if ay <= by:
            bottom_y, bottom_w = ay, wa
            top_y, top_w = by, wb
        else:
            bottom_y, bottom_w = by, wb
            top_y, top_w = ay, wa

        # 下方端點是有限點 => 檢查 py 不可小於它
        if bottom_w != 0:
            if py < bottom_y - epsilon:
                return False

        # 上方端點是有限點 => 檢查 py 不可大於它
        if top_w != 0:
            if py > top_y + epsilon:
                return False

        return True

    def clip_edge(self, VD: WingedEdge, edge_k: int, new_v_id: int, current_poly_side: str, s_left: tuple, s_right: tuple):
        """
        強制二選一修剪，解決浮點數誤差導致的「沒消掉」或「消錯」問題。
        """
        v_start = VD.start_vertex[edge_k]
        v_end = VD.end_vertex[edge_k]
        
        p_start = (VD.x_vertex[v_start], VD.y_vertex[v_start])
        p_end = (VD.x_vertex[v_end], VD.y_vertex[v_end])

        # 距離平方函式
        def dist_sq(p, s):
            return (p[0]-s[0])**2 + (p[1]-s[1])**2

        # 計算 "偏右程度" 分數
        # Score < 0 : 偏左 (靠近 s_left)
        # Score > 0 : 偏右 (靠近 s_right)
        score_start = dist_sq(p_start, s_left) - dist_sq(p_start, s_right)
        score_end   = dist_sq(p_end, s_left)   - dist_sq(p_end, s_right)

        target_vertex_to_replace = None

        if current_poly_side == 'LEFT':
            # 【左圖視角】
            # 要保留 "偏左" 的點 (Score 小的)
            # 要切掉 "偏右" 的點 (Score 大的)
            
            # 直接比較兩個端點，誰的分數比較大 (比較靠近右邊)，誰就被切掉
            # 這種 "相對比較" 能夠抵抗所有浮點數誤差
            if score_start > score_end:
                target_vertex_to_replace = 'START'
            else:
                target_vertex_to_replace = 'END'

        else: # RIGHT
            # 【右圖視角】
            # 要保留 "偏右" 的點 (Score 大的)
            # 要切掉 "偏左" 的點 (Score 小的)
            
            # 誰的分數比較小 (比較靠近左邊)，誰就被切掉
            if score_start < score_end:
                target_vertex_to_replace = 'START'
            else:
                target_vertex_to_replace = 'END'

        discarded_v_id = None

        # 執行替換
        # 被刪除的方向的邊是通往垃圾頂點的，因為已經被切斷連接了(其他指向的邊也會被清掉)
        if target_vertex_to_replace == 'START':
            discarded_v_id = VD.start_vertex[edge_k] # 記住這個即將被拋棄的點

            VD.start_vertex[edge_k] = new_v_id
            if VD.edge_around_vertex[new_v_id] is None:
                 VD.set_edge_around_vertex(new_v_id, edge_k)
        
        elif target_vertex_to_replace == 'END':
            discarded_v_id = VD.end_vertex[edge_k] # 記住這個即將被拋棄的點

            VD.end_vertex[edge_k] = new_v_id
            if VD.edge_around_vertex[new_v_id] is None:
                 VD.set_edge_around_vertex(new_v_id, edge_k)

        # DEBUG 印出被刪除的線
        if DEBUG:
            print(f"discard x, y = {VD.x_vertex[discarded_v_id]}, {VD.y_vertex[discarded_v_id]}")

        # ---------------------------------------------------------
        # 強制更新 Polygon 的入口指標
        # ---------------------------------------------------------
        # 為什麼要在這裡做？
        # 因為 edge_k 的其中一端剛剛被改接到縫合線上，它是絕對安全的「倖存者」。
        # 原本這兩個 Polygon 可能指向那些即將被丟棄的 "discarded_v_id" 連接的邊，
        # 現在強制把它們指向這條 edge_k，避免它們跟著幽靈邊一起陪葬。
        
        l_poly = VD.left_polygon[edge_k]
        r_poly = VD.right_polygon[edge_k]

        if l_poly is not None:
            VD.edge_around_polygon[l_poly] = edge_k
            
        if r_poly is not None:
            VD.edge_around_polygon[r_poly] = edge_k

    def remove_disconnected_components(self, VD: WingedEdge, seam_edges: list):
        """
        從縫合線 (Seam) 出發，遍歷所有可到達的邊。
        沒被訪問到的邊就是「斷裂的孤島 (Ghost Edges)」，統一刪除。
        
        Args:
            VD: WingedEdge 物件
            seam_edges: 這次合併產生的所有縫合線 ID 列表 (作為 BFS 起點)
        """
        if not seam_edges:
            return

        # MARK 階段：BFS 找出所有活著的邊
        visited = set()
        queue = list(seam_edges)
        
        # seam 已經在 queue 裡了，先把起點標記
        for e in seam_edges:
            visited.add(e)

        while queue:
            curr_e = queue.pop(0)
            
            neighbors = [
                VD.cw_successor[curr_e],
                VD.ccw_successor[curr_e],
                VD.cw_predecessor[curr_e],
                VD.ccw_predecessor[curr_e]
            ]
            
            for neighbor in neighbors:
                # 過濾無效指標 (-1, None, 或已訪問)
                if neighbor is not None and neighbor != -1 and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # 刪除沒被標記的邊
        total_edges = len(VD.start_vertex)
        deleted_count = 0
        
        for k in range(total_edges):
            # 如果這條邊存在 (不是 None)，但沒被 BFS 摸到 -> 它是幽靈邊
            if (VD.start_vertex[k] is not None or VD.start_vertex[k] != -1) and k not in visited:
                # 執行刪除 (Soft Delete)
                VD.start_vertex[k] = -1
                VD.end_vertex[k] = -1
                VD.left_polygon[k] = -1
                VD.right_polygon[k] = -1
                VD.cw_successor[k] = -1
                VD.ccw_successor[k] = -1
                VD.cw_predecessor[k] = -1
                VD.ccw_predecessor[k] = -1
                
                deleted_count += 1
                
        if DEBUG and deleted_count > 0:
            print(f"Mark-and-Sweep Cleaned {deleted_count} ghost edges.")

    def get_clipped_edge_segment(self, VD, edge_k):
        """
        因為 w=0 已經存了實體座標，這裡直接回傳端點座標即可。
        """
        v_start = VD.start_vertex[edge_k]
        v_end   = VD.end_vertex[edge_k]

        if v_start is None and v_end is None:
            return None, None
        
        if VD.x_vertex[v_start] is None or VD.x_vertex[v_end] is None:
            return None, None

        p1 = (VD.x_vertex[v_start], VD.y_vertex[v_start], VD.w_vertex[v_start])
        p2 = (VD.x_vertex[v_end],   VD.y_vertex[v_end],  VD.w_vertex[v_end])
        
        return p1, p2

    def upper_tangent(self, hull_left, hull_right):
        """
        回傳：(i_idx, j_idx)
        其中 i_idx 是 hull_left 裡的 index（不是 point index）
            j_idx 是 hull_right 裡的 index
        對應的點就是：
            points[ hull_left[i_idx] ], points[ hull_right[j_idx] ]
        """
        nL = len(hull_left)
        nR = len(hull_right)

        # i = max(range(nL),  key=lambda k: self.points[hull_left[k]][0])
        # j = min(range(nR),  key=lambda k: self.points[hull_right[k]][0])

        i = max(range(nL), key=lambda k: (self.points[hull_left[k]][0], self.points[hull_left[k]][1]))
        j = min(range(nR), key=lambda k: (self.points[hull_right[k]][0], self.points[hull_right[k]][1]))

        def orient(p, q, r):
            return (q[0]-p[0]) * (r[1]-p[1]) - (q[1]-p[1]) * (r[0]-p[0])

        changed = True
        while changed:
            changed = False

            # 固定 j，調整 i（沿左 hull）
            while True:
                i_next = (i - 1) % nL
                p_j  = self.points[hull_right[j]]
                p_i  = self.points[hull_left[i]]
                p_in = self.points[hull_left[i_next]]

                # 如果線段 (p_i, p_j) 還可以往上撐，則移動 i，這邊以畫布的 Y 軸向下為正方向
                if orient(p_j, p_i, p_in) > 0:
                    i = i_next
                    changed = True
                else:
                    break

            # 固定 i，調整 j（沿右 hull）
            while True:
                j_next = (j + 1) % nR
                p_i  = self.points[hull_left[i]]
                p_j  = self.points[hull_right[j]]
                p_jn = self.points[hull_right[j_next]]

                if orient(p_i, p_j, p_jn) < 0:
                    j = j_next
                    changed = True
                else:
                    break

        return i, j

    def lower_tangent(self, hull_left, hull_right):
        """
        作法類似，但方向跟不等號反過來。
        """
        nL = len(hull_left)
        nR = len(hull_right)

        # i = max(range(nL), key=lambda k: self.points[hull_left[k]][0])
        # j = min(range(nR), key=lambda k: self.points[hull_right[k]][0])
        # 取得左凸包中最右邊的點。X相等時，取Y最大的點。
        i = max(range(nL), key=lambda k: (self.points[hull_left[k]][0], self.points[hull_left[k]][1]))
        # 取得右凸包中最左邊的點。X相等時，取Y最小的點。
        j = min(range(nR), key=lambda k: (self.points[hull_right[k]][0], self.points[hull_right[k]][1]))

        def orient(p, q, r):
            return (q[0]-p[0]) * (r[1]-p[1]) - (q[1]-p[1]) * (r[0]-p[0])

        changed = True
        while changed:
            changed = False

            while True:
                i_next = (i + 1) % nL
                p_j  = self.points[hull_right[j]]
                p_i  = self.points[hull_left[i]]
                p_in = self.points[hull_left[i_next]]

                # 對下切線來說，要讓兩個 hull 都在「線的上方」
                if orient(p_j, p_i, p_in) < 0:
                    i = i_next
                    changed = True
                else:
                    break

            # 調整 j
            while True:
                j_next = (j - 1) % nR
                p_i  = self.points[hull_left[i]]
                p_j  = self.points[hull_right[j]]
                p_jn = self.points[hull_right[j_next]]

                if orient(p_i, p_j, p_jn) > 0:
                    j = j_next
                    changed = True
                else:
                    break

        return i, j

    def merge_structures(self, VD_L: WingedEdge, VD_R: WingedEdge) -> WingedEdge:
        merged = WingedEdge()

        # 複製左圖 (VD_L) - 索引不變
        merged.w_vertex.extend(VD_L.w_vertex)
        merged.x_vertex.extend(VD_L.x_vertex)
        merged.y_vertex.extend(VD_L.y_vertex)
        merged.edge_around_vertex.extend(VD_L.edge_around_vertex)

        merged.site_of_polygon.extend(VD_L.site_of_polygon)
        merged.edge_around_polygon.extend(VD_L.edge_around_polygon)

        merged.start_vertex.extend(VD_L.start_vertex)
        merged.end_vertex.extend(VD_L.end_vertex)
        merged.left_polygon.extend(VD_L.left_polygon)
        merged.right_polygon.extend(VD_L.right_polygon)
        merged.cw_predecessor.extend(VD_L.cw_predecessor)
        merged.ccw_predecessor.extend(VD_L.ccw_predecessor)
        merged.cw_successor.extend(VD_L.cw_successor)
        merged.ccw_successor.extend(VD_L.ccw_successor)

        # 計算偏移量 (Offsets)
        v_offset = len(VD_L.w_vertex)          # 頂點偏移
        p_offset = len(VD_L.edge_around_polygon) # 多邊形偏移
        e_offset = len(VD_L.start_vertex)      # 邊偏移

        # 複製右圖 (VD_R) - 加上偏移量
        merged.w_vertex.extend(VD_R.w_vertex)
        merged.x_vertex.extend(VD_R.x_vertex)
        merged.y_vertex.extend(VD_R.y_vertex)
        # 頂點查邊的索引也要偏移邊的 ID
        merged.edge_around_vertex.extend(
            [(k + e_offset) if k is not None else None for k in VD_R.edge_around_vertex]
        )

        merged.site_of_polygon.extend(VD_R.site_of_polygon)
        # 多邊形查邊的索引也要偏移
        merged.edge_around_polygon.extend(
            [(k + e_offset) if k is not None else None for k in VD_R.edge_around_polygon]
        )

        # 複製右圖的邊 - 所有參考都要偏移
        merged.start_vertex.extend(
            [(v + v_offset) if v is not None else None for v in VD_R.start_vertex]
        )
        merged.end_vertex.extend(
            [(v + v_offset) if v is not None else None for v in VD_R.end_vertex]
        )
        
        merged.left_polygon.extend(
            [(p + p_offset) if p is not None else None for p in VD_R.left_polygon]
        )
        merged.right_polygon.extend(
            [(p + p_offset) if p is not None else None for p in VD_R.right_polygon]
        )

        merged.cw_predecessor.extend(
            [(e + e_offset) if e is not None else None for e in VD_R.cw_predecessor]
        )
        merged.ccw_predecessor.extend(
            [(e + e_offset) if e is not None else None for e in VD_R.ccw_predecessor]
        )
        merged.cw_successor.extend(
            [(e + e_offset) if e is not None else None for e in VD_R.cw_successor]
        )
        merged.ccw_successor.extend(
            [(e + e_offset) if e is not None else None for e in VD_R.ccw_successor]
        )

        return merged, v_offset, p_offset, e_offset

    def trace_and_build_chain(self,
                          start_bisector_points,
                          end_bisector_points,
                          VD_L: WingedEdge, S_L: list,
                          VD_R: WingedEdge, S_R: list) -> WingedEdge:
        """
        第一版：不做真正的『追蹤分界鏈』，
        只建立一條由 start_bisector_points 的垂直平分線構成的無限長邊。
        回傳一個只包含這條邊的 WingedEdge。
        """
        HP = WingedEdge()

        pL, pR = start_bisector_points
        m, b = self.get_perpendicular_bisector(pL, pR)

        # 用很大的範圍來當端點
        if m is None:
            x = b
            v1 = HP.add_vertex(x, -1e6)
            v2 = HP.add_vertex(x,  1e6)
        else:
            x1, x2 = -1e6, 1e6
            y1 = m * x1 + b
            y2 = m * x2 + b
            v1 = HP.add_vertex(x1, y1)
            v2 = HP.add_vertex(x2, y2)

        # polygon id 先留 None，之後縫合時會處理
        k = HP.add_edge(
            start_v=v1,
            end_v=v2,
            left_p=None,
            right_p=None,
            cw_pred=None,
            ccw_pred=None,
            cw_succ=None,
            ccw_succ=None
        )

        # 兩端點的 edge_around_vertex 先指自己
        HP.set_edge_around_vertex(v1, k)
        HP.set_edge_around_vertex(v2, k)

        return HP

    def clip_segment_to_canvas(self, x1, y1, x2, y2, width, height):
        """
        把線段 (x1, y1)-(x2, y2) 裁切到畫布 [0,width] x [0,height] 內。
        回傳 (cx1, cy1, cx2, cy2) 或 None (完全不在畫布內)
        """
        x_min, x_max = 0.0, float(width)
        y_min, y_max = 0.0, float(height)

        dx = x2 - x1
        dy = y2 - y1

        # Liang–Barsky
        p = [-dx, dx, -dy, dy]
        q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]

        t1, t2 = 0.0, 1.0  # 參數 t ∈ [t1, t2] 是裁切後的有效範圍

        for pi, qi in zip(p, q):
            if pi == 0:
                # 線段平行於這個邊界，如果一開始就超出範圍 → 完全在外面
                if qi < 0:
                    return None
                # 否則這個邊界不影響裁切
            else:
                r = qi / pi
                if pi < 0:
                    # 進入區間的限制：t ≥ r
                    if r > t2:
                        return None
                    if r > t1:
                        t1 = r
                else:
                    # 離開區間的限制：t ≤ r
                    if r < t1:
                        return None
                    if r < t2:
                        t2 = r

        cx1 = x1 + t1 * dx
        cy1 = y1 + t1 * dy
        cx2 = x1 + t2 * dx
        cy2 = y1 + t2 * dy

        return cx1, cy1, cx2, cy2

    def draw_voronoi_from_winged(self, W: WingedEdge, color='blue', tag='voronoi'):
        num_edges = len(W.start_vertex)

        # 劃出 edge
        for k in range(num_edges):
            sj = W.start_vertex[k]
            ej = W.end_vertex[k]

            if sj == None or ej == None:
                continue

            x1, y1 = W.x_vertex[sj], W.y_vertex[sj]
            x2, y2 = W.x_vertex[ej], W.y_vertex[ej]

            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue

            self.lines.append( ( (x1, y1), (x2, y2) ) )  # 這邊直接使用原本的self.lines比較方便儲存檔案

            # 畫線
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill=color,
                width=1,
                tags=tag
            )

            if DEBUG:
                clipped = self.clip_segment_to_canvas(x1, y1, x2, y2, self.width, self.height)
                if clipped is None: continue

                cx1, cy1, cx2, cy2 = clipped
                mid_x = (cx1 + cx2) / 2
                mid_y = (cy1 + cy2) / 2
            
                self.canvas.create_text(
                    mid_x, mid_y,
                    text=str(k),
                    fill="red",
                    font=("Arial", 10),
                    tags=tag
                )
        
        if DEBUG:
            for i in range(len(W.edge_around_polygon)):
                idx = W.site_of_polygon[i]
                x, y = self.points[idx]

                self.canvas.create_text(
                    x+10, y,
                    text=str(i),
                    fill="blue",
                    font=("Arial", 10),
                    tags=tag
                )

            for j in range(len(W.x_vertex)):
                if j is None: continue
                x = W.x_vertex[j]
                y = W.y_vertex[j]

                if x is None or y is None: continue

                self.canvas.create_text(
                    x, y,
                    text=str(j),
                    fill="purple",
                    font=("Arial", 15),
                    tags=tag
                )

    def clear_canvas_step(self):
        self.canvas.delete('voronoi_step')
        self.canvas.delete('convex_hull')
        self.canvas.delete('point_step')
        self.canvas.delete('hyperplane')

    def step_by_step(self, snapshot):
        """
        根據快照類型繪製當前狀態
        snapshot 結構: {'type': str, 'vd': WingedEdge, 'info': list/dict}
        # 順序 ['left_hull', 'right_hull', 'left_voronoi', right_voronoi', 'merged_hull', 'seam_edges', 'remove_line', 'merged_voronoi']
        merged_hull 和 remove_line 搭配沒有消線的 merged_voronoi
        """

        step_type = snapshot['type']
        vd = snapshot['vd']
        info = snapshot['info'] # 這裡是 Convex Hull 的 index 列表，或是 seam edge id

        if step_type == 'left_hull':
            self.draw_hull_lines(info, color='orange', tag='convex_hull')
            self.draw_point_step(vd, 'blue', tag='point_step')
        elif step_type == 'right_hull':
            self.draw_hull_lines(info, color='orange', tag='convex_hull')
            self.draw_point_step(vd, 'red', tag='point_step')
        elif step_type == 'left_voronoi':
            self.draw_voronoi_from_winged(vd, color='blue', tag='voronoi_step')
        elif step_type == 'right_voronoi':
            self.draw_voronoi_from_winged(vd, color='red', tag='voronoi_step')
        elif step_type == 'merged_hull':
            self.canvas.delete('convex_hull')
            self.draw_hull_lines(info, color='orange')
        elif step_type == 'seam_edges':
            seam_edges = info 
            if isinstance(seam_edges, list):
                for edge_k in seam_edges:
                    self.draw_single_edge(vd, edge_k, color='purple', width=3)
        elif step_type == 'remove_line':
            self.canvas.delete('voronoi_step')
            self.canvas.delete('convex_hull')
            self.draw_voronoi_from_winged(vd, color='green', tag='voronoi_step')
        elif step_type == 'merged_voronoi':
            self.clear_canvas_step()
            self.draw_voronoi_from_winged(vd, color='black', tag='voronoi')
    
    def draw_point_step(self, W: WingedEdge, color, tag='point_step', size=3):
        for idx in W.site_of_polygon:
            x, y = self.points[idx]
            self.draw_point(x, y, color, tag, size)

    def draw_hull_lines(self, hull_indices, color='orange', dash=(8, 2), tag='convex_hull'):
        if not hull_indices or len(hull_indices) < 2: return
        
        # 畫出凸包多邊形
        pts = []
        for idx in hull_indices:
            p = self.points[idx]
            pts.append(p[0])
            pts.append(p[1])
        
        # 加上起點形成閉合
        p_start = self.points[hull_indices[0]]
        pts.append(p_start[0])
        pts.append(p_start[1])

        self.canvas.create_line(pts, fill=color, width=2, dash=dash, tags=tag)

    def draw_single_edge(self, vd, k, color='green', width=2, tag='hyperplane'):
        if k is None or k == -1: return
        start_v = vd.start_vertex[k]
        end_v = vd.end_vertex[k]
        
        if start_v is not None and end_v is not None:
            x1, y1 = vd.x_vertex[start_v], vd.y_vertex[start_v]
            x2, y2 = vd.x_vertex[end_v], vd.y_vertex[end_v]
            
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, tags=tag)

    def draw_voronoi(self, color='black'):
        # 清除舊的 Voronoi 線 (保留點)
        if not self.points: return

        self.canvas.delete("voronoi")
        self.lines = []

        # 永遠不會改變 self.points 的順序，並且我可以使用預先儲存在 self.point_index 的 mapping 來找到點的全域 index
        sorted_points = sorted(self.points, key=lambda p: (p[0], p[1]))

        print("Building Voronoi Diagram using Divide and Conquer...")
        self.voronoi, _ = self.build_voronoi_dc(sorted_points)

        self.draw_voronoi_from_winged(self.voronoi, color=color)

if __name__ == "__main__":
    main_root = tk.Tk()
    
    app = VoronoiDiagram(main_root)
    
    main_root.mainloop()