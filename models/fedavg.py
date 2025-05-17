import copy
import torch
import torch.nn as nn
from tqdm import tqdm

from models.utils.federated_model import FederatedModel


class FedAvG(FederatedModel):
    NAME = 'fedavg'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedAvG, self).__init__(nets_list,args,transform)

    def ini(self, same_for_all = True):
        self.global_net = copy.deepcopy(self.nets_list[0])
        if same_for_all:
            global_w = self.nets_list[0].state_dict()
            for _, net in enumerate(self.nets_list):
                net.load_state_dict(global_w)


    def loc_update(self,priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
        self.aggregate_nets(None)

        return  None

    # def _train_net(self, index, net, train_loader):
    #
    #     print(f'Training client {index}')
    #     net.fit(train_loader)
    #     # optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
    #     # criterion = nn.CrossEntropyLoss()
    #     # criterion.to(self.device)
    #     # iterator = tqdm(range(self.local_epoch))
    #     # for _ in iterator:
    #     #     for batch_idx, (images, labels) in enumerate(train_loader):
    #     #         images = images.to(self.device)
    #     #         labels = labels.to(self.device)
    #     #         outputs = net(images)
    #     #         loss = criterion(outputs, labels)
    #     #         optimizer.zero_grad()
    #     #         loss.backward()
    #     #         iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index,loss)
    #     #         optimizer.step()

    def _train_net(self, index, model, train_loader, device="cpu", reg_ratio=0.5, verbose=True):
        data_loader = model.prepare_data(train_loader)
        model = model.to(device)
        model.train()
        model.initialize_optimizer()  # initializes if not already done
        optimizer = model.optimizer
        criterion = nn.MSELoss()

        iterator = tqdm(range(self.args.local_epoch))
        for _ in iterator:
            epoch_loss = 0
            for xb, ryb, yb, fyb in data_loader:
                xb = xb.to(device)
                ryb = ryb.to(device)
                yb = yb.to(device)
                fyb = fyb.to(device)

                optimizer.zero_grad()
                out_ry, out_y, out_fy, latent = model(xb)

                loss_ry = criterion(out_ry, torch.squeeze(ryb))
                loss_y = criterion(out_y, yb)
                loss_fy = criterion(out_fy, torch.squeeze(fyb))

                loss = (reg_ratio / 2) * loss_ry + (1 - reg_ratio) * loss_y + (reg_ratio / 2) * loss_fy
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if verbose:
                iterator.set_description(f"Local Participant {index} - Loss: {epoch_loss:.4f}")

            model.fit_history.append(epoch_loss)

        model.total_local_epoch += self.args.local_epoch