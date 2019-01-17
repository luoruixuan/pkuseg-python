import pkuseg.model
from typing import List

import pkuseg.inference as _inf
import pkuseg.data


def get_grad_SGD_minibatch(
    grad: List[float], model: pkuseg.model.Model, X: List[pkuseg.data.Example]
):
    # if idset is not None:
    #     idset.clear()
    all_id_set = set()
    errors = 0
    for x in X:
        error, id_set = get_grad_CRF(grad, model, x)
        errors += error
        all_id_set.update(id_set)

    return errors, all_id_set


def get_grad_CRF(
    grad: List[float], model: pkuseg.model.Model, x: pkuseg.data.Example
):

    id_set = set()

    n_tag = model.n_tag
    bel = _inf.belief(len(x), n_tag)
    belMasked = _inf.belief(len(x), n_tag)

    Ylist, YYlist, maskYlist, maskYYlist = _inf.getYYandY(model, x)
    # print(YYlist.shape, Ylist.shape, maskYYlist.shape, maskYlist.shape)
    _inf.get_beliefs(bel, model, x, Ylist, YYlist)
    _inf.get_beliefs(belMasked, model, x, maskYlist, maskYYlist)

    ZGold = belMasked.Z
    Z = bel.Z

    for i, node_feature_list in enumerate(x.features):
        # node_feature_list = x.features[i]

        for feature_id in node_feature_list:
            for tag_id in range(n_tag):
                trans_id = model._get_node_tag_feature_id(feature_id, tag_id)
                id_set.add(tag_id)
                grad[trans_id] += bel.belState[i][tag_id]
                grad[trans_id] -= belMasked.belState[i][tag_id]

    for i in range(1, len(x)):
        for tag_id in range(n_tag):
            for pre_tag_id in range(n_tag):
                trans_id = model._get_tag_tag_feature_id(pre_tag_id, tag_id)
                id_set.add(trans_id)

                grad[trans_id] += bel.belEdge[i][pre_tag_id, tag_id]
                grad[trans_id] -= belMasked.belEdge[i][pre_tag_id, tag_id]

    return Z - ZGold, id_set
