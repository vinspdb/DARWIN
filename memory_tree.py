def pruned_tree1(check_case_id, hashtable):
    if check_case_id.is_root:
        return
    result = dict((new_val, new_k) for new_k, new_val in hashtable.items()).get(check_case_id)
    if len(check_case_id.children) ==1 and result == None:
        pruned_tree1(check_case_id.parent, hashtable)
        check_case_id.parent = None

def pruned_tree(check_case_id, hashtable):
    if check_case_id.is_root:
        return
    result = dict((new_val, new_k) for new_k, new_val in hashtable.items()).get(check_case_id)
    if check_case_id.is_leaf and result == None:
        pruned_tree1(check_case_id.parent, hashtable)
        check_case_id.parent = None


def check_exist_child(node, name):
        for n in node:
            if name == n.id:
               return n

def get_trace(node):
        list_act = []
        if node.is_root == True:
            return list_act
        else:
            list_act = get_trace(node.parent)
            list_act.append(node.id)
            return list_act
