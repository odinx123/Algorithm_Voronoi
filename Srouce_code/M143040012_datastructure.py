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