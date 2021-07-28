import re
from pytorch_model_summary import summary


def model_summary(model, inputs, print_summary=False, max_depth=1, show_parent_layers=False):
    # _ = summary(model, x_in, print_summary=True)
    kwargs = {'max_depth': max_depth,
              'show_parent_layers': show_parent_layers}
    sT = summary(model, inputs, show_input=True, print_summary=False, **kwargs)
    sF = summary(model, inputs, show_input=False, print_summary=False, **kwargs)

    st = sT.split('\n')
    sf = sF.split('\n')

    sf1 = re.split(r'\s{2,}', sf[1])
    out_i = sf1.index('Output Shape')

    ss = []
    i_esc = []
    for i in range(0, len(st)):
        if len(re.split(r'\s{2,}', st[i])) == 1:
            ssi = st[i]
            if len(set(st[i])) == 1:
                i_esc.append(i)
        else:
            sfi = re.split(r'\s{2,}', sf[i])
            sti = re.split(r'\s{2,}', st[i])
            
            ptr = st[i].index(sti[out_i]) + len(sti[out_i])
            in_1 = sf[i].index(sfi[out_i-1]) + len(sfi[out_i-1])
            in_2 = sf[i].index(sfi[out_i]) + len(sfi[out_i])
            ssi = st[i][:ptr] + sf[i][in_1:in_2] + st[i][ptr:]
        ss.append(ssi)

    n_str = max([len(s) for s in ss])
    for i in i_esc:
        ss[i] = ss[i][-1] * n_str

    ss = '\n'.join(ss)
    if print_summary:
        print(ss)

    return ss
