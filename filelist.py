import os


def fnames(dir):
    return sorted([k[0:-4] for k in os.listdir(dir) if not k.startswith(".")])


def fnames2(dir):
    fn = [k[0 : len(k) - 4] for k in os.listdir(dir) if not k.startswith(".")]
    ind = fn[0].rfind("_")
    dic = {}
    for k in range(len(fn)):
        dic[int(fn[k][ind + 1 :])] = fn[k]
    a = [dic[k] for k in sorted(dic.keys())]
    print(a)
    return a


def segdir(imgdir):
    return os.path.join(
        imgdir[: imgdir.rfind(os.path.sep)],
        "cleans" + imgdir[imgdir.rfind(os.path.sep) + 5 :],
    )
