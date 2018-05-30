import itertools
import pandas as pd

class FPNode(object):
    def __init__(self, name, value, children, parent):
        self.name = name
        self.value = value
        self.children = children
        self.parent = parent
        self.next = None

    def __repr__(self):
        return "{}: {}".format(self.name, self.value)

    def has_child(self, name):
        for child in self.children:
            if child.name == name:
                return True
        return False

    def get_child(self, name):
        for child in self.children:
            if child.name == name:
                return child
        return None

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def link_next(self, node):
        nxt = self
        while nxt.next is not None:
            nxt = nxt.next
        nxt.next = node


class FPTree(object):

    def __init__(self, mini_support):
        self.header_table = []
        self.rootNode = FPNode(name='root', value=1, children=[], parent=None)
        self.items_dic = {}
        self.mini_support = mini_support

    def build_tree(self, input_data):
        self.__build_items_dic(input_data)
        self.__build_header_table()
        tree_data = self.filter_raw_input(input_data)
        # print(tree_data)
        # tree_data = [['f', 'a', 'b'], ['f', 'a', 'c'], ['a', 'b'], ['c']]
        self.create_tree_with_data(tree_data)
        # print(self.rootNode)

    '''
    private method 
    build a header_table
    '''
    def __build_items_dic(self, input_data):
        for line in input_data:
            for word in line:
                if self.items_dic.get(word) is None:
                    self.items_dic[word] = 0
                self.items_dic[word] += 1
        # print(self.items_dic)

    '''
    private build header_table with items
    '''
    def __build_header_table(self):
        self.header_table = [FPNode(name=key, value=val, children=[], parent=None)
                             for key,val in self.items_dic.items()
                             if val >= self.mini_support
                             ]
        self.header_table.sort(key=lambda x: x.value, reverse=True)
        # print(self.header_table)

    def filter_raw_input(self, input_data):
        # print(self.items_dic)
        tree_data = [sorted([word for word in line if self.items_dic[word] >= self.mini_support]
                            , key=lambda x:self.items_dic[x], reverse=True)
                     for line in input_data
                     ]

        return tree_data

    def create_tree_with_data(self, tree_data):
        for line in tree_data:
            self.insert_tree(line)

    def insert_tree(self, insert_nodes):
        cur_node = self.rootNode
        i_of_insert_nodes = 0
        len_of_insert_nodes = len(insert_nodes)
        while i_of_insert_nodes < len_of_insert_nodes and \
                cur_node.has_child(insert_nodes[i_of_insert_nodes]):
            cur_node = cur_node.get_child(insert_nodes[i_of_insert_nodes])
            cur_node.value += 1
            i_of_insert_nodes += 1

        for i in range(i_of_insert_nodes, len_of_insert_nodes):
            new_node_name = insert_nodes[i]
            new_node = FPNode(name=new_node_name, value=1, children=[], parent=cur_node)
            cur_node.add_child(new_node)
            self.update_header_table(new_node)
            cur_node = new_node

    def update_header_table(self, node):
        for nd in self.header_table:
            if nd.name == node.name:
                nd.link_next(node)

    def generate_patterns(self, path, item):
        patterns = {}
        keys = path.keys()
        for i in range(1, len(path)+1):
            p_list = list(itertools.combinations(keys, i))
            # print('list', list(p_list))
            ptn_lst = [sorted(list(x)+[item.name]) for x in p_list]
            # print(ptn_lst)
            # set((list(x)).append(item.name))
            for x in ptn_lst:
                patterns[tuple(x)] = min([path[key] for key in x if key != item.name])

        patterns[tuple([item.name])] = item.value
        return patterns

    def find_prefix_path(self, item):
        single_ptn = {}
        item_val = item.value
        item = item.parent
        while item.name != 'root':
            single_ptn[item.name] = item_val
            item = item.parent
        return single_ptn

    def mine_patterns(self):
        patterns = {}
        for item in self.header_table:
            cur = item.next
            single_path = {}
            while cur is not None:
                temp_dic = self.find_prefix_path(cur)
                single_path = merge_two_dic(single_path, temp_dic)
                cur = cur.next
            single_path = {k: v for k, v in single_path.items() if v >= self.mini_support}
            single_pattern = self.generate_patterns(single_path, item)
            patterns = merge_two_dic(patterns, single_pattern)
        return patterns


def merge_two_dic(dic1, dic2):
    dic1 = dict(dic1)
    merged_dic = dic1.copy()
    merged_dic.update(dic2)
    for key in merged_dic.keys():
        merged_dic[key] = dic1.get(key, 0) + dic2.get(key, 0)
    return merged_dic


'''
modify from https://github.com/waxnkw/fp-growth/blob/master/pyfpgrowth/pyfpgrowth.py 
'''
def generate_association_rules(patterns, confidence_threshold):
    rules = []
    for itemset in patterns.keys():
        upper_support = patterns[itemset]

        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in patterns:
                    lower_support = patterns[antecedent]
                    confidence = float(upper_support) / lower_support

                    if confidence >= confidence_threshold \
                            and (antecedent.__contains__('republican0') or antecedent.__contains__('democrat0')):
                        rules.append([list(antecedent), list(consequent)])
    return rules


if __name__ == '__main__':

    df = pd.read_csv('./A.csv')
    tree = FPTree(150)
    tree.build_tree(df.values)
    p = tree.mine_patterns()
    print(len(p))
    x = generate_association_rules(p, 0.9)
    print(len(x))
