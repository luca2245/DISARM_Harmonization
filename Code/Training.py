from Utils import *
from Load_Dataset import *
from DISARM_class import *
from tensorboardX import SummaryWriter



class Saver():
  def __init__(self, opts):
    self.display_dir = os.path.join(opts.display_dir, opts.name)
    self.model_dir = os.path.join(opts.result_dir, opts.name)
    self.image_dir = os.path.join(self.model_dir, 'images')
    self.display_freq = opts.display_freq
    self.img_save_freq = opts.img_save_freq
    self.model_save_freq = opts.model_save_freq

    # make directory
    if not os.path.exists(self.display_dir):
        os.makedirs(self.display_dir)
    if not os.path.exists(self.model_dir):
        os.makedirs(self.model_dir)
    if not os.path.exists(self.image_dir):
        os.makedirs(self.image_dir)

    # create tensorboard writer
    self.writer = SummaryWriter(log_dir=self.display_dir)

  # write losses and images to tensorboard
  def write_display(self, total_it, model):
    if (total_it + 1) % self.display_freq == 0:
        # write loss
        members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
        for m in members:
            self.writer.add_scalar(m, getattr(model, m), total_it)
        # write img
        image_dis = torchvision.utils.make_grid(model.image_display, nrow=model.image_display.size(0)//2)/2 + 0.5
        self.writer.add_image('Image', image_dis, total_it)

  # save result images
  def write_img(self, ep, model):
    if (ep + 1) % self.img_save_freq == 0:
        assembled_images1, assembled_images2, assembled_images3 = model.assemble_outputs()
        img_filename1 = '%s/gen_%05d_slice_1.jpg' % (self.image_dir, ep)
        img_filename2 = '%s/gen_%05d_slice_2.jpg' % (self.image_dir, ep)
        img_filename3 = '%s/gen_%05d_slice_3.jpg' % (self.image_dir, ep)
        pil_image1 = Image.fromarray(assembled_images1.squeeze(), mode='L')
        pil_image2 = Image.fromarray(assembled_images2.squeeze(), mode='L')
        pil_image3 = Image.fromarray(assembled_images3.squeeze(), mode='L')
        pil_image1.save(img_filename1)
        pil_image2.save(img_filename2)
        pil_image3.save(img_filename3)
    elif ep == -1:
        assembled_images1, assembled_images2, assembled_images3 = model.assemble_outputs()
        img_filename1 = '%s/gen_last_slice_1.jpg' % (self.image_dir, ep)
        img_filename2 = '%s/gen_last_slice_2.jpg' % (self.image_dir, ep)
        img_filename3 = '%s/gen_last_slice_3_.jpg' % (self.image_dir, ep)
        pil_image1 = Image.fromarray(assembled_images1.squeeze(), mode='L')
        pil_image2 = Image.fromarray(assembled_images2.squeeze(), mode='L')
        pil_image3 = Image.fromarray(assembled_images3.squeeze(), mode='L')
        pil_image1.save(img_filename1)
        pil_image2.save(img_filename2)
        pil_image3.save(img_filename3)

  # save model
  def write_model(self, ep, total_it, model):
    if (ep + 1) % self.model_save_freq == 1:
        print('--- save the model @ ep %d ---' % (ep))
        model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
    elif ep == -1:
        model.save('%s/last.pth' % self.model_dir, ep, total_it)



def train(opts):
    
    print('\n--- load dataset ---')
    dataset = data_multi_aug(opts) # Loader with augmentation (random elastic deformation)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
    
    print('\n--- set saver ---')
    saver = Saver(opts)
    
    
    # LOAD MODEL
    print('\n--- load model ---')
    model = DISARM(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d'%(ep0))
    
    # TRAINING
    print('\n--- train ---')
    max_it = 100000
    
    stop_training = False

    for ep in range(ep0, opts.n_ep):
        for it, (images, c_org) in enumerate(train_loader):
            if images.size(0) != opts.batch_size:
                continue

            # input data
            images = images.cuda(opts.gpu).detach()
            c_org = c_org.cuda(opts.gpu).detach()

            # update model
            if opts.isDcontent:
                if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                    model.update_D_content(images, c_org)
                    continue
                else:
                    model.update_D(images, c_org)
                    model.update_EG()
            else:
                model.update_D(images, c_org)
                model.update_EG()
        
      
            if (total_it+1) % opts.img_save_freq == 0:
                saver.write_img(-1, model)
            if (total_it+1) % opts.model_save_freq == 0:
                saver.write_model(-1,total_it, model)
            print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(-1,total_it, model)
                stop_training = True  # Set the flag to stop training
                break
      
    

        if stop_training:
            break  # Exit the outer loop

    # decay learning rate
    if opts.n_ep_decay > -1:
          model.update_lr()
    
    # save result image
    saver.write_img(ep, model)
    
    # Save network weights
    saver.write_model(ep, total_it, model)


