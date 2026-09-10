import numpy as np
import pytest
import torch

from biomaster.odti_v3_incremental import IncrementalConfig, IncrementalV3, measured_query_loss


@pytest.mark.parametrize('evidence,pretrained', [(False,False),(True,False),(True,True)])
def test_zero_checkpoint_preserves_both_frozen_directions(evidence, pretrained):
    model=IncrementalV3(IncrementalConfig(hidden_dim=12,width=8,evidence=evidence,pretrained=pretrained)).eval()
    base=torch.tensor([-3.,0.,2.,7.])
    output=model(base,torch.randn(4,12),torch.rand(4,8),torch.randn(4,768),torch.randn(4,1280))
    assert torch.equal(output,base[:,None].expand(-1,2))


def test_direct_evidence_signs_and_missing_branch():
    model=IncrementalV3(IncrementalConfig(hidden_dim=4,width=8,evidence=True)).eval()
    with torch.no_grad():
        model.positive_weight.fill_(1)
        model.negative_weight.fill_(1)
    hidden=torch.zeros(4,4)
    support=torch.zeros(4,8)
    support[1,1]=.7;support[1,5]=1
    support[2,0]=.7;support[2,4]=1
    support[3,0:2]=.9  # Both branches masked; padded similarity has no effect.
    out=model(torch.zeros(4),hidden,support)
    assert (out[1]>0).all() and (out[2]<0).all()
    assert (out[[0,3]]==0).all()


def test_zero_evidence_weights_can_learn():
    model=IncrementalV3(IncrementalConfig(hidden_dim=4,width=8,evidence=True))
    support=torch.zeros(1,8);support[0,1]=.8;support[0,5]=1
    (-model(torch.zeros(1),torch.zeros(1,4),support).sum()).backward()
    assert (model.positive_weight.grad < 0).all()


def test_unknown_padding_is_ignored_in_loss_and_gradient():
    scores=torch.tensor([[1.,-1.,float('nan')]],requires_grad=True)
    labels=torch.tensor([[1.,0.,float('nan')]])
    loss=measured_query_loss(scores,labels,torch.tensor([[True,True,False]]),.5)
    reference=measured_query_loss(scores[:,:2],labels[:,:2],torch.ones(1,2,dtype=torch.bool),.5)
    assert torch.allclose(loss,reference)
    loss.backward()
    assert torch.isfinite(scores.grad).all() and scores.grad[0,2]==0


def test_pairwise_gradient_improves_correct_order_and_respects_queries():
    scores=torch.zeros(2,2,requires_grad=True)
    labels=torch.tensor([[1.,0.],[0.,1.]])
    measured_query_loss(scores,labels,torch.ones_like(scores,dtype=torch.bool),1.).backward()
    assert (scores.grad[labels==1]<0).all() and (scores.grad[labels==0]>0).all()
