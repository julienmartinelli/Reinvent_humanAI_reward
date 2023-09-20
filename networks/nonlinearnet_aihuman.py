import torch
import torch.nn as nn


class NonLinearNetDefer(nn.Module):
    def __init__(self, num_features):
        super(NonLinearNetDefer, self).__init__()

        # define architecture for classifier 1
        self.classifier1 = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(
                1024, 1
            ),  # output layer with one neuron for binary classification
            nn.Sigmoid(),  # sigmoid activation for probability output (g_y)
        )

        # define architecture for classifier 2 or human model (similar to classifier 1)
        self.classifier2 = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),  # sigmoid activation for probability output (g_h)
        )

        # define the architecture for the rejector
        self.decision_classifier = nn.Sequential(
            nn.Linear(
                2, 3
            ),  # input size is 2: takes in as input the outputs of classifier 1 and classifier 2
            nn.Softmax(dim=1)  # softmax activation for probability distribution
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


def optimization_loop(num_epochs, optimizer, model, X, y, h, criterion):

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        combined_outputs, decision_outputs = model(X)

        # losses for classifier 1 and classifier 2
        loss_clf1 = criterion(combined_outputs[:, 0].unsqueeze(1), y)
        loss_clf2 = criterion(combined_outputs[:, 1].unsqueeze(1), h)

        boolean = (
            (decision_outputs[:, -1] > decision_outputs[:, 0])
            * (decision_outputs[:, 1] > decision_outputs[:, 0])
            * 1.0
        )
        outputs = (
            boolean * combined_outputs[:, 1] + (1 - boolean) * combined_outputs[:, 0]
        ).unsqueeze(1)
        loss_system = criterion(outputs, y)
        # combine the losses
        total_loss = loss_clf1 + loss_clf2 + loss_system

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}")
