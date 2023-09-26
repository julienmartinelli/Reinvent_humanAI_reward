import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch import optim


class NonLinearNetDeferSoftmax(nn.Module):
    def __init__(self, num_features, dropout):
        super(NonLinearNetDeferSoftmax, self).__init__()

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
                2, 3
            ),  # input size is 2: takes in as input the outputs of classifier 1 and classifier 2
            nn.Softmax()
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


def optimization_loop(
    num_epochs, optimizer, model, X, X_h, y, h, criterion, alpha, active_learning=False
):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        combined_outputs, decision_outputs = model(X)
        combined_outputs_h, decision_outputs_h = model(X_h)

        # losses for clf 1 and system based on y
        loss_clf1 = criterion(combined_outputs[:, 0].unsqueeze(1), y) / len(X)
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
        loss_system = criterion(outputs, y) / len(X)

        # losses for clf 2 and system based on h
        loss_clf2 = criterion(combined_outputs_h[:, 1].unsqueeze(1), h) / len(X_h)
        # boolean_h = (h == 1).reshape(-1) * (
        #     decision_outputs_h[:, 0] > combined_outputs_h[:, 0]
        # ) * (combined_outputs_h[:, 1] > combined_outputs_h[:, 0]) + (h == 0).reshape(
        #     -1
        # ) * (
        #     decision_outputs_h[:, 0] > combined_outputs_h[:, 0]
        # ) * (
        #     combined_outputs_h[:, 1] < combined_outputs_h[:, 0]
        # ) * 1.0

        # # still experimental... Same criterion as loss_system above, but with h.
        # # underlying assumption being that for h close to y this is helpful.
        # outputs_h = (
        #     boolean_h * combined_outputs_h[:, 1]
        #     + (1 - boolean_h) * combined_outputs_h[:, 0]
        # ).unsqueeze(1)
        # loss_system_h = criterion(outputs_h, h)

        # combine the losses
        # total_loss = loss_clf1 + loss_clf2 + loss_system + loss_system_h
        total_loss = alpha * (loss_clf1 + loss_clf2) + (1 - alpha) * (
            loss_system  # + loss_system_h
        )

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}")


def optimization_loop_softmax(num_epochs, optimizer, model, X, X_h, y, h, criterion):

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        combined_outputs, decision_outputs = model(X)
        combined_outputs_h, decision_outputs_h = model(X_h)

        # losses for clf 1 and system based on y
        loss_clf1 = criterion(combined_outputs[:, 0].unsqueeze(1), y)
        boolean = (
            (decision_outputs[:, -1] > decision_outputs[:, 0])
            * (decision_outputs[:, 1] > decision_outputs[:, 0])
            * 1.0
        )
        outputs = (
            boolean * combined_outputs[:, 1] + (1 - boolean) * combined_outputs[:, 0]
        ).unsqueeze(1)
        loss_system = criterion(outputs, y)

        # losses for clf 2 and system based on h
        loss_clf2 = criterion(combined_outputs_h[:, 1].unsqueeze(1), h)
        boolean_h = (
            (decision_outputs_h[:, -1] > decision_outputs_h[:, 0])
            * (decision_outputs_h[:, 1] > decision_outputs_h[:, 0])
            * 1.0
        )
        # still experimental... Same criterion as loss_system above, but with h.
        # underlying assumption being that for h close to y this is helpful.
        outputs_h = (
            boolean_h * combined_outputs_h[:, 1]
            + (1 - boolean_h) * combined_outputs_h[:, 0]
        ).unsqueeze(1)
        loss_system_h = criterion(outputs_h, h)

        # combine the losses
        total_loss = (loss_clf1 + loss_clf2) + (loss_system + loss_system_h)

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}")


def get_prob_distribution(
    models, X
):  # requires running X through multiple forward passes to generate a distribution
    X = torch.tensor(X, dtype=torch.float32)
    prob_dist = [m(X) for m in models]
    prob_dist = np.stack(prob_dist, axis=1)
    prob_dist = torch.tensor(prob_dist)

    return prob_dist


def get_uncertainty_scores(model, X):
    features = torch.tensor(features, dtype=torch.float32)
    probs, _ = model(X)
    prob1_h = probs[:, 1]
    prob0_h = 1 - probs[:, 1]
    scores = []

    for i in range(len(prob1_h)):
        scores.append(
            -np.sum(
                [prob0_h[i] * np.log2(prob0_h[i]), prob1_h[i] * np.log2(prob1_h[i])]
            )
        )
    return scores


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
    criterion,
    active_learning=False,
):

    best_alpha, best_f1 = 0, 0
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
            criterion,
            alpha,
            active_learning=False,
        )
        final_pred, _, _, _, _ = test_time_prediction(model, X_val)
        f1 = f1_score(final_pred, y_val)
        print(alpha, f1)
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
            criterion,
            best_alpha,
            active_learning=False,
        )
    return best_alpha, model


def test_time_prediction(model, X_test):

    model.eval()
    combined_outputs, decision_outputs = model(X_test)
    pred_clf = (combined_outputs > 0.5).float()
    # boolean = torch.tensor((decision_outputs[:, 0] > combined_outputs[:, 0]) * 1., dtype=torch.float32)
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
