import json


NUM_STRS = {str(i) for i in range(10)}

def has_num(vname):
    for chr in vname:
        if chr in NUM_STRS:
            return True
    return False

def split_num(vname):
    results = []
    cursor = 0
    for idx, chr in enumerate(vname):
        if chr in NUM_STRS:
            results.append(vname[cursor:idx])
            cursor = idx+1
    results.append(vname[cursor:])
    return [i for i in results if i]


class Name2Label:

    def __init__(self):
        with open("resources/fixed_lookup.json") as f:
            self.lookup = json.load(f)
        with open("resources/lemma.json") as f:
            self.lemma = json.load(f)
        with open("resources/specials.txt") as f:
            self.specials = set([l for l in f.read().splitlines() if l])
        with open("resources/vocabs.txt") as f:
            self.vocabs = set([l for l in f.read().splitlines() if l])
            self.vocabs |= set(self.lookup.values())
        self.bad_names = set()
    
    def name_to_label(self, vname):
        if self._base_check(vname.lower()) is not None:
            return [self._base_check(vname.lower())]
        vanme_items = self._spilt_name(vname)
        labels = []

        for item in vanme_items:
            if len(item) == 0:
                continue

            lower_item = item.lower()

            if self._base_check(lower_item):
                labels.append(self._base_check(lower_item))
            
            elif lower_item[0] == "<" and lower_item[-1] == ">":
                labels.append(lower_item)

            # elif lower_item in {"i", "j", "k", "n"}:
            #     labels.append("<counter>")
            elif len(lower_item) == 1:
                continue
            
            elif lower_item[-1] == "s" and lower_item[:-1] in self.lookup:
                labels.append(self.lookup[lower_item[:-1]])
            
            elif has_num(lower_item):
                for sub_item in split_num(lower_item):
                    labels += self.name_to_label(sub_item)
            
            elif self._base_check(lower_item[1:]):
                labels.append(self._base_check(lower_item[1:]))
            
            elif self._base_check(lower_item[:-1]):
                labels.append(self._base_check(lower_item[:-1]))
            
            elif lower_item[:2] in ['yy', 'is', 'fd'] and self._base_check(lower_item[2:]):
                labels += [lower_item[:2], self._base_check(lower_item[2:])]
                # print(vname, labels)
            
            elif lower_item[-2:] in ['fd'] and self._base_check(lower_item[:-2]):
                labels += [lower_item[-2:], self._base_check(lower_item[:-2])]
                # print(vname, labels)

            elif len(lower_item) > 4:
                parse_result = self._parse_dirty_vname_new(lower_item)
                if parse_result:
                    labels += parse_result
        
        if labels == []:
            if len(vanme_items) == 1:
                if vanme_items[0] in ['i', 'j', 'n', 'x', 'y', 'q', 'p']:
                    labels = vanme_items
                elif len(vanme_items[0]) == 1:
                    labels = ['<single>']
                elif len(vanme_items[0]) == 2:
                    labels = vanme_items
            
        if labels == []:
            self.bad_names.add(vname)
        return labels

    def _base_check(self, lower_item):
        labels = None
        if lower_item in self.specials:
            labels = lower_item
        elif lower_item in self.lookup:
            labels = self.lookup[lower_item]
        elif lower_item in self.lemma:
            labels = self.lemma[lower_item]
        elif lower_item in self.vocabs:
            labels = lower_item
        if labels is None:
            _ = labels
        return labels
    
    # def _parse_dirty_vname(self, vname, depth=1):
        
    #     if depth > 4: return
    #     if len(vname) <= 2: return
            
    #     for idx in range(len(vname)):

    #         # cursor = idx
    #         # cursor = len(vname) - idx
    #         cursor = idx + int(len(vname)/2) if \
    #             idx < int(len(vname)/2) else len(vname) - idx

    #         sub_a = vname[:cursor]
    #         sub_b = vname[cursor:]
            
    #         sub_a = self._base_check(sub_a)
    #         sub_b = self._base_check(sub_b)

    #         if sub_a is not None and sub_b is not None:
    #             return [sub_a, sub_b]

    #         cond_a = self._parse_dirty_vname(sub_a, depth+1)

    #         if cond_a != None and sub_b is not None:
    #             return cond_a.append(sub_b)
            
    #         cond_b = self._parse_dirty_vname(sub_b, depth+1)

    #         if sub_a is not None and cond_b:
    #             return [sub_a] + cond_b
            
    #         if cond_a and cond_b:
    #             return cond_a + cond_b

    #     return 
    
    def _split_camel_name(self, maybe_camel_vname):
        results = []
        cursor = 0
        for idx, char in enumerate(maybe_camel_vname):
            if char.isupper():
                results.append(maybe_camel_vname[cursor:idx])
                cursor = idx
        results.append(maybe_camel_vname[cursor:])
        results = [i for i in results if i]
        return results
    
    def _spilt_name(self, vname):

        results = []
        cursor = 0
        for idx, char in enumerate(vname):
            if char in {"_", ".", '/', '-', '@', '>', '<', '*'}:
                if len(vname[cursor:idx]) > 15 and not vname[cursor:idx].islower():
                    results += self._split_camel_name(vname[cursor:idx])
                else:
                    results.append(vname[cursor:idx])
                cursor = idx+1
        results.append(vname[cursor:])
        results = [i for i in results if i]

        if len(results) > 1 or len(results) == 0:
            return results
        
        # maybe camel case varaiable name
        maybe_camel_vname = results[0]
        results = self._split_camel_name(maybe_camel_vname)
        return results
    
    def _parse_dirty_vname_new(self, vname):
        all_cases = set()

        for idx in range(len(vname)):

            if idx < 2 or len(vname) - idx < 2:
                continue
            all_cases.add((vname[:idx], vname[idx:]))

            for idx2 in range(idx, len(vname)):
                if idx2 - idx < 3 or len(vname) - idx2 < 3:
                    continue
                all_cases.add((vname[:idx], vname[idx:idx2], vname[idx2:]))
            
            for idx3 in range(idx):
                if idx3 < 3 or idx - idx3 < 3:
                    continue
                all_cases.add((vname[:idx3], vname[idx3:idx], vname[idx:]))
        
        candidates = []
        for item in all_cases:
            condidate = [self._base_check(x) for x in item]
            if all(condidate):
                candidates.append(condidate)

        candidates = sorted(candidates, key=lambda x: len(x))
        return candidates[0] if candidates else None


N2L = Name2Label()


def split_var_name(var_name):
    if var_name:
        return N2L.name_to_label(var_name)
    else:
        return []


if __name__=="__main__":
    name2label = Name2Label()
    print(name2label.name_to_label('lbuf_putc'))
    print(name2label.name_to_label('lucky_num'))
    print(split_var_name('lbuf_putc'))
    print(split_var_name('lucky_num'))
    print(split_var_name(None))
    print(split_var_name('__stack_chk_guard_ptr'))
