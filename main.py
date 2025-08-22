
import base64
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests
from PIL import Image
import io
from dotenv import load_dotenv, find_dotenv

from google.oauth2 import service_account
import google.auth
from google.auth.transport.requests import Request, AuthorizedSession

_ = load_dotenv(find_dotenv())

class VirtualTryOnAPI:
    
    def __init__(
        self, 
        location: str = "us-central1"
    ):
        """
        Args:
            project_id: Google Cloud プロジェクトID
            location: リージョン（デフォルト: us-central1）
            service_account_path: サービスアカウントJSONファイルのパス（オプション）
        """
        SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
        self.credentials, self.project_id = google.auth.default(scopes=SCOPES)
        print(f"使用中のプロジェクトID: {self.project_id}")

        self.location = location
        self.model_id = "virtual-try-on-preview-08-04"
        self.endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{location}/publishers/google/models/{self.model_id}:predict"


    def try_on(
        self,
        person_image_path: str,
        product_image_path: str,
        output_dir: str = "./output",
        # 必須パラメータ
        sample_count: int = 1,
        base_steps: int = 32,
        # オプションパラメータ
        add_watermark: bool = False,
        person_generation: str = "allow_all",
        safety_setting: str = "block_only_high",
        seed: Optional[int] = None,
        output_mime_type: str = "image/png",
        compression_quality: Optional[int] = None
    ) -> List[str]:
        """
        Virtual Try-Onを実行して画像を生成・保存
        
        Args:
            person_image_path: 人物画像のパス
            product_image_path: 商品画像のパス
            output_dir: 出力ディレクトリ
            
            === APIパラメータ ===
            sample_count: 生成する画像の数（1-4）
            base_steps: 画像生成のステップ数
            add_watermark: 透かしを追加するか（デフォルト: False）
            person_generation: 人物生成モード（"dont_allow", "allow_adult", "allow_all"）
            safety_setting: 安全フィルタ（"block_low_and_above", "block_medium_and_above", "block_only_high", "block_none"）
            seed: 乱数シード
            storage_uri: Cloud Storageの出力パス（オプション）
            output_mime_type: 出力形式（"image/png" or "image/jpeg"）
            compression_quality: JPEG圧縮品質（1-100、JPEGの場合のみ）
        
        Returns:
            保存された画像のパスリスト
        """
        
        # 画像の存在確認
        if not os.path.exists(person_image_path):
            raise FileNotFoundError(f"人物画像が見つかりません: {person_image_path}")
        if not os.path.exists(product_image_path):
            raise FileNotFoundError(f"商品画像が見つかりません: {product_image_path}")
        
        # 画像をBase64エンコード
        def encode_image(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        
        person_base64 = encode_image(person_image_path)
        product_base64 = encode_image(product_image_path)
        
        # リクエストボディ作成
        request_body = {
            "instances": [{
                "personImage": {
                    "image": {"bytesBase64Encoded": person_base64}
                },
                "productImages": [{
                    "image": {"bytesBase64Encoded": product_base64}
                }]
            }],
            "parameters": {
                "sampleCount": sample_count,
                "baseSteps": base_steps,
                "addWatermark": add_watermark,
                "personGeneration": person_generation,
                "safetySetting": safety_setting,
                "outputOptions": {
                    "mimeType": output_mime_type
                }
            }
        }
        
        # オプションパラメータの追加
        if seed is not None:
            request_body["parameters"]["seed"] = seed
        if compression_quality and output_mime_type == "image/jpeg":
            request_body["parameters"]["outputOptions"]["compressionQuality"] = compression_quality
        
        # APIリクエスト
        print(f"Virtual Try-On実行中... (画像数: {sample_count}, ステップ数: {base_steps})")
        authed = AuthorizedSession(self.credentials)
        response = authed.post(
            self.endpoint,
            json=request_body
        )
        
        if response.status_code != 200:
            raise Exception(f"APIエラー: {response.status_code} - {response.text}")
        
        # 画像の保存
        result = response.json()
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, prediction in enumerate(result.get("predictions", [])):
            # Base64デコード
            img_data = base64.b64decode(prediction["bytesBase64Encoded"])
            img = Image.open(io.BytesIO(img_data))
            
            # ファイル名生成
            ext = ".png" if prediction.get("mimeType") == "image/png" else ".jpg"
            timestamp = Path(person_image_path).stem + "_" + Path(product_image_path).stem
            filepath = Path(output_dir) / f"vton_{timestamp}_{i}{ext}"
            
            # 保存
            img.save(filepath)
            saved_paths.append(str(filepath))
            print(f"✅ 画像を保存: {filepath}")
        
        return saved_paths


if __name__ == "__main__":

    # 最小限の使用例
    api = VirtualTryOnAPI()
    
    results = api.try_on(
        person_image_path="./images/person.png",
        product_image_path="./images/dress.png",
        output_dir="./results",
        # APIパラメータ
        sample_count=1,              
        base_steps=32,               
        add_watermark=False,         
        seed=42,                    
        output_mime_type="image/png", 
    )

    print("生成完了しました")
