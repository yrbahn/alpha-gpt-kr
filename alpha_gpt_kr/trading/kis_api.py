"""
한국투자증권 KIS OpenAPI 클라이언트
"""

import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


class KISApi:
    """한국투자증권 OpenAPI 클라이언트"""
    
    # API 엔드포인트
    BASE_URL_REAL = "https://openapi.koreainvestment.com:9443"  # 실전투자
    BASE_URL_VIRT = "https://openapivts.koreainvestment.com:29443"  # 모의투자
    
    # 토큰 캐시 디렉토리
    TOKEN_CACHE_DIR = Path.home() / ".kis_tokens"
    
    def __init__(
        self,
        app_key: str,
        app_secret: str,
        account_no: str,
        is_real: bool = False
    ):
        """
        Args:
            app_key: API Key (앱 키)
            app_secret: API Secret (앱 시크릿)
            account_no: 계좌번호 (8자리-2자리)
            is_real: True=실전투자, False=모의투자
        """
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_no = account_no
        self.is_real = is_real
        
        self.base_url = self.BASE_URL_REAL if is_real else self.BASE_URL_VIRT
        self.access_token = None
        self.token_expires_at = None
        
        # 토큰 캐시 디렉토리 생성
        self.TOKEN_CACHE_DIR.mkdir(exist_ok=True)
        
        # 캐시된 토큰 로드
        self._load_cached_token()
        
        logger.info(f"KIS API 초기화: {'실전' if is_real else '모의'}투자")
        
    def _get_token_cache_path(self) -> Path:
        """토큰 캐시 파일 경로"""
        mode = "real" if self.is_real else "virt"
        return self.TOKEN_CACHE_DIR / f"token_{mode}_{self.app_key[:8]}.json"
    
    def _load_cached_token(self):
        """캐시된 토큰 로드"""
        cache_path = self._get_token_cache_path()
        
        if not cache_path.exists():
            return
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # 만료 시간 확인
            expires_at = datetime.fromisoformat(data['expires_at'])
            
            if datetime.now() < expires_at:
                self.access_token = data['access_token']
                self.token_expires_at = expires_at
                logger.info(f"✅ 캐시된 토큰 로드 성공 (만료: {expires_at.strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                logger.info("⏰ 캐시된 토큰 만료됨, 재발급 필요")
                
        except Exception as e:
            logger.warning(f"토큰 캐시 로드 실패: {e}")
    
    def _save_token_cache(self, token: str, expires_in: int = 86400):
        """토큰 캐시 저장 (기본: 24시간)"""
        cache_path = self._get_token_cache_path()
        
        expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        data = {
            'access_token': token,
            'expires_at': expires_at.isoformat(),
            'created_at': datetime.now().isoformat()
        }
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        
        self.token_expires_at = expires_at
        logger.info(f"💾 토큰 캐시 저장 (만료: {expires_at.strftime('%Y-%m-%d %H:%M:%S')})")
    
    def _get_headers(self, tr_id: str) -> Dict[str, str]:
        """API 요청 헤더 생성"""
        # 토큰 확인 및 갱신
        if not self.access_token or (self.token_expires_at and datetime.now() >= self.token_expires_at):
            self._get_access_token()
            
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }
    
    def _get_access_token(self) -> str:
        """접근 토큰 발급"""
        logger.info("🔑 새로운 Access Token 발급 중...")
        
        url = f"{self.base_url}/oauth2/tokenP"
        
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        self.access_token = result['access_token']
        
        # 토큰 캐시 저장 (24시간 유효)
        expires_in = result.get('expires_in', 86400)  # 기본 24시간
        self._save_token_cache(self.access_token, expires_in)
        
        logger.info("✅ Access Token 발급 완료")
        return self.access_token
    
    def get_balance(self) -> Dict:
        """계좌 잔고 조회"""
        # 실전투자: TTTC8434R, 모의투자: VTTC8434R
        tr_id = "TTTC8434R" if self.is_real else "VTTC8434R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        
        # 계좌번호 파싱 (예: 12345678-01)
        acct_parts = self.account_no.split('-')
        
        params = {
            "CANO": acct_parts[0],  # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": acct_parts[1],  # 계좌상품코드 뒤 2자리
            "AFHR_FLPR_YN": "N",  # 시간외단일가여부
            "OFL_YN": "",  # 오프라인여부
            "INQR_DVSN": "02",  # 조회구분(01:대출일별, 02:종목별)
            "UNPR_DVSN": "01",  # 단가구분
            "FUND_STTL_ICLD_YN": "N",  # 펀드결제분포함여부
            "FNCG_AMT_AUTO_RDPT_YN": "N",  # 융자금액자동상환여부
            "PRCS_DVSN": "01",  # 처리구분
            "CTX_AREA_FK100": "",  # 연속조회검색조건
            "CTX_AREA_NK100": ""  # 연속조회키
        }
        
        headers = self._get_headers(tr_id)
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        result = response.json()
        
        if result['rt_cd'] != '0':
            raise Exception(f"잔고 조회 실패: {result['msg1']}")
        
        logger.info("✅ 계좌 잔고 조회 완료")
        
        # output2가 리스트인 경우 첫 번째 요소 반환
        output2 = result['output2']
        if isinstance(output2, list) and len(output2) > 0:
            return output2[0]
        return output2
    
    def get_holdings(self) -> List[Dict]:
        """보유 종목 조회"""
        tr_id = "TTTC8434R" if self.is_real else "VTTC8434R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        
        acct_parts = self.account_no.split('-')
        
        params = {
            "CANO": acct_parts[0],
            "ACNT_PRDT_CD": acct_parts[1],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        headers = self._get_headers(tr_id)
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        result = response.json()
        
        if result['rt_cd'] != '0':
            raise Exception(f"보유종목 조회 실패: {result['msg1']}")
        
        holdings = result['output1']
        logger.info(f"✅ 보유 종목 조회: {len(holdings)}개")
        return holdings
    
    def get_current_price(self, ticker: str) -> Dict:
        """현재가 조회"""
        tr_id = "FHKST01010100"  # 주식현재가 시세
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 시장분류코드 (J:주식)
            "FID_INPUT_ISCD": ticker  # 종목코드
        }
        
        headers = self._get_headers(tr_id)
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        result = response.json()
        
        if result['rt_cd'] != '0':
            raise Exception(f"현재가 조회 실패: {result['msg1']}")
        
        return result['output']
    
    def order_stock(
        self,
        ticker: str,
        qty: int,
        order_type: str = "01",  # 00:지정가, 01:시장가
        side: str = "buy"  # buy or sell
    ) -> Dict:
        """주식 주문"""
        # 실전투자 매수:TTTC0802U, 매도:TTTC0801U
        # 모의투자 매수:VTTC0802U, 매도:VTTC0801U
        if self.is_real:
            tr_id = "TTTC0802U" if side == "buy" else "TTTC0801U"
        else:
            tr_id = "VTTC0802U" if side == "buy" else "VTTC0801U"
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        
        acct_parts = self.account_no.split('-')
        
        # 현재가 조회
        current_price = self.get_current_price(ticker)
        price = current_price['stck_prpr']  # 주식현재가
        
        data = {
            "CANO": acct_parts[0],
            "ACNT_PRDT_CD": acct_parts[1],
            "PDNO": ticker,  # 종목코드
            "ORD_DVSN": order_type,  # 주문구분 (00:지정가, 01:시장가)
            "ORD_QTY": str(qty),  # 주문수량
            "ORD_UNPR": "0" if order_type == "01" else price  # 주문단가 (시장가=0)
        }
        
        headers = self._get_headers(tr_id)
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        if result['rt_cd'] != '0':
            raise Exception(f"주문 실패: {result['msg1']}")
        
        logger.info(f"✅ {side.upper()} 주문 완료: {ticker} {qty}주")
        return result['output']
    
    def buy_stock(self, ticker: str, qty: int, order_type: str = "01") -> Dict:
        """매수 주문"""
        return self.order_stock(ticker, qty, order_type, "buy")
    
    def sell_stock(self, ticker: str, qty: int, order_type: str = "01") -> Dict:
        """매도 주문"""
        return self.order_stock(ticker, qty, order_type, "sell")
