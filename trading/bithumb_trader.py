import hashlib
import hmac
import time
import requests
import urllib.parse

class BithumbTrader:
    """빗썸 API 연동을 위한 클래스"""
    
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.bithumb.com"

    def _create_signature(self, endpoint, params):
        """API 요청 서명 생성"""
        nonce = str(int(time.time() * 1000))
        query_string = urllib.parse.urlencode(params)
        message = endpoint + chr(0) + query_string + chr(0) + nonce
        signature = hmac.new(self.secret_key.encode('utf-8'), message.encode('utf-8'), hashlib.sha512).hexdigest()
        return signature, nonce

    def get_balance(self, currency="BTC"):
        """보유 자산 조회"""
        endpoint = "/info/balance"
        params = {"currency": currency}
        signature, nonce = self._create_signature(endpoint, params)

        headers = {
            "Api-Key": self.api_key,
            "Api-Sign": signature,
            "Api-Nonce": nonce
        }

        response = requests.post(self.base_url + endpoint, headers=headers, data=params)
        return response.json()

    def place_order(self, symbol, side, amount, price=None):
        """주문 실행 (시장가 또는 지정가)"""
        endpoint = "/trade/place"
        params = {
            "order_currency": symbol,
            "payment_currency": "KRW",
            "units": amount,
            "type": "market" if price is None else "limit"
        }
        if price:
            params["price"] = price  # 지정가 주문

        signature, nonce = self._create_signature(endpoint, params)
        headers = {
            "Api-Key": self.api_key,
            "Api-Sign": signature,
            "Api-Nonce": nonce
        }

        response = requests.post(self.base_url + endpoint, headers=headers, data=params)
        return response.json()

    def get_market_price(self, symbol="BTC_KRW"):
        """현재가 조회"""
        try:
            response = requests.get(f"{self.base_url}/public/ticker/{symbol}")
            data = response.json()
            return float(data["data"]["closing_price"])
        except Exception as e:
            print(f"실시간 가격 가져오기 실패: {e}")
            return None
