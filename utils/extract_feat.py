import open_clip


print("loading OpenCLIP model...")
open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', cache_dir="/kaiming-fast-vol/workspace/open_clip_model/")
open_clip_model.cuda().eval()

print("extracting 3D shape feature...")
xyz, feat = load_ply("demo/pc.ply")
shape_feat = model(xyz, feat, device='cuda', quantization_size=config.model.voxel_size) 

print("extracting text features...")
texts = ["owl", "chicken", "penguin"]
text_feat = extract_text_feat(texts, open_clip_model)
print("texts: ", texts)
print("3D-text similarity: ", F.normalize(shape_feat, dim=1) @ F.normalize(text_feat, dim=1).T)
