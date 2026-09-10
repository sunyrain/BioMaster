import torch
from biomaster.old_relation_retrieval import known_relation_loss


def test_all_known_relations_receive_positive_weight():
    scores=torch.zeros(1,3,requires_grad=True)
    known_relation_loss(scores,torch.tensor([[True,True,False]])).backward()
    assert (scores.grad[0,:2]<0).all() and scores.grad[0,2]>0


def test_empty_queries_do_not_become_negative_training_examples():
    scores=torch.tensor([[1.,0.],[100.,-100.]],requires_grad=True)
    labels=torch.tensor([[True,False],[False,False]])
    loss=known_relation_loss(scores,labels)
    assert torch.allclose(loss,known_relation_loss(scores[:1],labels[:1]))
    loss.backward()
    assert (scores.grad[1]==0).all()


def test_candidate_permutation_preserves_multirelation_loss():
    scores=torch.tensor([[3.,1.,2.],[4.,5.,1.]])
    known=torch.tensor([[1,0,1],[0,1,0]],dtype=torch.bool)
    permutation=torch.tensor([2,0,1])
    assert torch.allclose(known_relation_loss(scores,known),known_relation_loss(scores[:,permutation],known[:,permutation]))
