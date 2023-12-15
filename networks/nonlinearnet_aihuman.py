import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch import optim


class NonLinearNetDefer(nn.Module):
    def __init__(self, num_features, dropout):
        super(NonLinearNetDefer, self).__init__()

        # define architecture for classifier 1
        self.classifier1 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),  # output layer with one neuron for binary classification
            nn.Sigmoid(),  # sigmoid activation for probability output (g_y)
        )

        # define architecture for classifier 2 or human model (similar to classifier 1)
        self.classifier2 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),  # sigmoid activation for probability output (g_h)
        )

        # define the architecture for the rejector
        self.decision_classifier = nn.Sequential(
            nn.Linear(
                2, 1
            ),  # input size is 2: takes in as input the outputs of classifier 1 and classifier 2
            nn.Sigmoid()
            # nn.Softmax(dim=1)  # softmax activation for probability distribution
            # in this way we have 3 output heads [g_y, g_h, g_perp]
        )

    def forward(self, x):
        # forward pass for classifier 1
        out1 = self.classifier1(x)

        # forward pass for classifier 2
        out2 = self.classifier2(x)

        # combine the outputs of classifier 1 and classifier 2
        combined_output = torch.cat((out1, out2), dim=1)

        # forward pass for the rejector to decide which classifier predicts
        final_output = self.decision_classifier(combined_output)

        # return combined_output from classifier1 and classifier 2 trained separately
        # and final output which is the probability distribution p(y=1,h=1,r=1)
        return combined_output, final_output


class ClassifierSimple(nn.Module):
    def __init__(self, num_features, dropout):
        super(ClassifierSimple, self).__init__()

        # define architecture for classifier 1
        self.classifier1 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),  # output layer with one neuron for binary classification
            nn.Sigmoid(),  # sigmoid activation for probability output (g_y)
        )

    def forward(self, x):
        # forward pass for classifier 1
        out = self.classifier1(x)

        return out


def optimization_loop(
    num_epochs, optimizer, model, X, X_h, y, h, criterion_classifier, criterion_decision, alpha
):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        combined_outputs, decision_outputs = model(X)
        combined_outputs_h, decision_outputs_h = model(X_h)

        # losses for clf 1 and system based on y
        loss_clf1 = criterion_classifier(combined_outputs[:, 0].unsqueeze(1), y)
        # when label is 1 and human is more confident than classifier
        boolean = (y == 1).reshape(-1) * (
            decision_outputs[:, 0] > combined_outputs[:, 0]
        ) * (combined_outputs[:, 1] > combined_outputs[:, 0]) + (y == 0).reshape(-1) * (
            decision_outputs[:, 0] > combined_outputs[:, 0]
        ) * (
            combined_outputs[:, 1] < combined_outputs[:, 0]
        ) * 1.0
        # then defer decision should be true
        outputs = (
            boolean * combined_outputs[:, 1] + (1 - boolean) * combined_outputs[:, 0]
        ).unsqueeze(1)
        loss_system = criterion_decision(outputs, y)

        # losses for clf 2 and system based on h
        loss_clf2 = criterion_classifier(combined_outputs_h[:, 1].unsqueeze(1), h)
        boolean_h = (h == 1).reshape(-1) * (
             decision_outputs_h[:, 0] > combined_outputs_h[:, 0]
         ) * (combined_outputs_h[:, 1] > combined_outputs_h[:, 0]) + (h == 0).reshape(
             -1
         ) * (
             decision_outputs_h[:, 0] > combined_outputs_h[:, 0]
         ) * (
             combined_outputs_h[:, 1] < combined_outputs_h[:, 0]
         ) * 1.0

        # still experimental... Same criterion as loss_system above, but with h.
        # underlying assumption being that for h close to y this is helpful.
        outputs_h = (
             boolean_h * combined_outputs_h[:, 1]
             + (1 - boolean_h) * combined_outputs_h[:, 0]
         ).unsqueeze(1)
        loss_system_h = criterion_decision(outputs_h, h)

        # combine the losses
        # total_loss = loss_clf1 + loss_clf2 + loss_system + loss_system_h
        total_loss = alpha * (loss_clf1 + loss_clf2) + (1 - alpha) * (
            loss_system  + loss_system_h
        )

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}, Deferral: {boolean.mean().item()}")


def get_prob_distribution(
    model, features, num_passes=100, classifier_only=False
):
    model.eval()
    
    forward_passes_outputs = []

    with torch.no_grad():
        for i in range(num_passes):
            model.train()
            if classifier_only:
                preds = model(torch.tensor(features, dtype=torch.float32))
                forward_passes_outputs.append(
                [1-preds[:, 0].cpu().detach().numpy(),
                 preds[:, 0].cpu().detach().numpy()]
                 )
            else:
                preds, _ = model(torch.tensor(features, dtype=torch.float32))
                forward_passes_outputs.append(
                [1-preds[:, 1].cpu().detach().numpy(),
                 preds[:, 1].cpu().detach().numpy()]
                 )
            
    
    prob_dist = torch.tensor(forward_passes_outputs, dtype=torch.float32).T  # to have shape (num_samples, num_passes)

    return prob_dist


def optimize_alpha(
    alpha_grid,
    lr,
    num_features,
    dropout,
    num_epochs,
    X_val,
    y_val,
    X,
    X_h,
    y,
    h,
    criterion_classifier,
    criterion_decision
):

    best_alpha, best_f1, F1s = 0, 0, []
    for alpha in alpha_grid:
        model = NonLinearNetDefer(num_features, dropout)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        optimization_loop(
            num_epochs,
            optimizer,
            model,
            X,
            X_h,
            y,
            h,
            criterion_classifier,
            criterion_decision,
            alpha
        )
        final_pred, _, _, _, _ = test_time_prediction(model, X_val)
        f1 = f1_score(final_pred, y_val)
        print(alpha, f1)
        F1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha

    # final round for the optimal alpha)
    if len(alpha_grid) > 1:
        print("final retraining with best alpha")
        model = NonLinearNetDefer(num_features, dropout)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        optimization_loop(
            num_epochs,
            optimizer,
            model,
            X,
            X_h,
            y,
            h,
            criterion_classifier,
            criterion_decision,
            best_alpha
        )
    return best_alpha, model, F1s


def test_time_prediction(model, X_test):

    model.eval()
    combined_outputs, decision_outputs = model(X_test)
    pred_clf = (combined_outputs > 0.5).float()
    boolean = (
        (decision_outputs[:, 0] > combined_outputs[:, 0])
        * (
            (combined_outputs[:, 0] > 0.5)
            * (combined_outputs[:, 1] > combined_outputs[:, 0])
            + (combined_outputs[:, 0] < 0.5)
            * (combined_outputs[:, 1] < combined_outputs[:, 0])
        )
    ).float()
    final_predictions = (boolean * pred_clf[:, 1]) + (1 - boolean) * pred_clf[:, 0]
    return final_predictions, pred_clf, boolean, combined_outputs, decision_outputs
