"""
LeetCode API Client for fetching problems and submitting solutions
Uses session cookies from browser to bypass Cloudflare protection
"""
import time
import json
import uuid
import cloudscraper
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from config import Config


class LeetCodeClient:
    """Client for interacting with LeetCode API using session cookies"""
    
    def __init__(self, session_cookie: str = None, csrf_token: str = None):
        # Get session cookies from parameters or config
        self.session_cookie = session_cookie or Config.LEETCODE_SESSION_COOKIE
        self.csrf_token = csrf_token or Config.LEETCODE_CSRF_TOKEN
        
        # Use cloudscraper to bypass Cloudflare protection
        self.session = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'darwin',  # macOS
                'mobile': False
            }
        )
        
        # Set session cookies if provided
        if self.session_cookie and self.csrf_token:
            self._set_session_cookies()
        
    def _set_session_cookies(self):
        """Set cookies from browser session"""
        self.session.cookies.set('LEETCODE_SESSION', self.session_cookie, domain='.leetcode.com')
        self.session.cookies.set('csrftoken', self.csrf_token, domain='.leetcode.com')
        print(f"✓ Session cookies configured")
    
    def login(self) -> bool:
        """
        Verify session cookies are valid
        No username/password login - only uses session cookies
        """
        if not self.session_cookie or not self.csrf_token:
            print(f"✗ Session cookies not configured")
            print(f"   Please set LEETCODE_SESSION_COOKIE and LEETCODE_CSRF_TOKEN in .env")
            return False
        
        try:
            # Test if cookies are valid by making a simple GraphQL request
            query = """
            query globalData {
              userStatus {
                username
                isSignedIn
              }
            }
            """
            
            response = self._graphql_request(query, {})
            
            if response and 'data' in response:
                user_status = response.get('data', {}).get('userStatus', {})
                if user_status.get('isSignedIn'):
                    username = user_status.get('username', 'unknown')
                    print(f"✓ Successfully authenticated as {username}")
                    return True
                else:
                    print(f"✗ Session cookies are invalid or expired")
                    print(f"   Please update cookies in .env file")
                    return False
            else:
                print(f"✗ Failed to verify session cookies")
                return False
                
        except Exception as e:
            print(f"✗ Authentication error: {str(e)}")
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_problem_list(self, limit: int = 50, skip: int = 0, 
                        difficulty: Optional[str] = None) -> List[Dict]:
        """
        Fetch list of problems from LeetCode using V2 API
        
        Args:
            limit: Number of problems to fetch
            skip: Number of problems to skip
            difficulty: Filter by difficulty (EASY, MEDIUM, HARD)
        
        Returns:
            List of problem metadata
        """
        query = """
        query problemsetQuestionListV2($filters: QuestionFilterInput, $limit: Int, $searchKeyword: String, $skip: Int, $sortBy: QuestionSortByInput, $categorySlug: String) {
          problemsetQuestionListV2(
            filters: $filters
            limit: $limit
            searchKeyword: $searchKeyword
            skip: $skip
            sortBy: $sortBy
            categorySlug: $categorySlug
          ) {
            questions {
              id
              titleSlug
              title
              questionFrontendId
              paidOnly
              difficulty
              topicTags {
                name
                slug
              }
              acRate
            }
            totalLength
            hasMore
          }
        }
        """
        
        # Build filters in V2 format
        filters = {
            "filterCombineType": "ALL",
            "statusFilter": {"questionStatuses": [], "operator": "IS"},
            "difficultyFilter": {"difficulties": [], "operator": "IS"},
            "languageFilter": {"languageSlugs": [], "operator": "IS"},
            "topicFilter": {"topicSlugs": [], "operator": "IS"},
            "acceptanceFilter": {},
            "frequencyFilter": {},
            "frontendIdFilter": {},
            "lastSubmittedFilter": {},
            "publishedFilter": {},
            "companyFilter": {"companySlugs": [], "operator": "IS"},
            "positionFilter": {"positionSlugs": [], "operator": "IS"},
            "contestPointFilter": {"contestPoints": [], "operator": "IS"},
            "premiumFilter": {"premiumStatus": [], "operator": "IS"}
        }
        
        if difficulty:
            filters["difficultyFilter"]["difficulties"] = [difficulty]
        
        variables = {
            "categorySlug": "all-code-essentials",
            "limit": limit,
            "skip": skip,
            "filters": filters,
            "searchKeyword": "",
            "sortBy": {"sortField": "CUSTOM", "sortOrder": "ASCENDING"}
        }
        
        response = self._graphql_request(query, variables, operation_name="problemsetQuestionListV2")
        
        if response and 'data' in response and 'problemsetQuestionListV2' in response['data']:
            return response['data']['problemsetQuestionListV2']['questions']
        return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_problem_details(self, title_slug: str) -> Optional[Dict]:
        """
        Get detailed problem information including description and code template
        
        Args:
            title_slug: URL slug of the problem (e.g., 'two-sum')
        
        Returns:
            Problem details dictionary
        """
        query = """
        query questionData($titleSlug: String!) {
          question(titleSlug: $titleSlug) {
            questionId
            questionFrontendId
            title
            titleSlug
            content
            difficulty
            likes
            dislikes
            categoryTitle
            topicTags {
              name
              slug
            }
            codeSnippets {
              lang
              langSlug
              code
            }
            sampleTestCase
            exampleTestcases
            hints
          }
        }
        """
        
        variables = {"titleSlug": title_slug}
        response = self._graphql_request(query, variables, operation_name="questionData")
        
        if response and 'data' in response and response['data']['question']:
            problem = response['data']['question']
            
            # Extract Python code template
            python_template = None
            for snippet in problem['codeSnippets']:
                if snippet['langSlug'] == 'python3':
                    python_template = snippet['code']
                    break
            
            return {
                'question_id': problem['questionFrontendId'],
                'title': problem['title'],
                'title_slug': problem['titleSlug'],
                'difficulty': problem['difficulty'],
                'description': problem['content'],
                'code_template': python_template,
                'topics': [tag['name'] for tag in problem['topicTags']],
                'sample_test': problem.get('sampleTestCase', ''),
                'hints': problem.get('hints', []),
                'likes': problem.get('likes', 0),
                'dislikes': problem.get('dislikes', 0)
            }
        
        return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def submit_solution(self, title_slug: str, code: str, question_id: str = None, lang: str = 'python3') -> Optional[str]:
        """
        Submit a solution to LeetCode using REST API
        
        Args:
            title_slug: Problem URL slug
            code: Solution code
            question_id: Question ID (e.g., "1" for two-sum)
            lang: Programming language
        
        Returns:
            Submission ID if successful
        """
        # If question_id not provided, we need to extract it from problem data
        # For now, we'll need to get it from the problem details
        url = f"{Config.LEETCODE_BASE_URL}/problems/{title_slug}/submit/"
        
        payload = {
            "lang": lang,
            "question_id": question_id or "0",  # Will need to get this from problem details
            "typed_code": code
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Referer': f'{Config.LEETCODE_BASE_URL}/problems/{title_slug}/',
            'Origin': Config.LEETCODE_BASE_URL,
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'X-CSRFToken': self.csrf_token,
            'random-uuid': str(uuid.uuid4()),
        }
        
        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 429:
                # Rate limited - raise exception to trigger retry
                print(f"⚠ Rate limited (429) - will retry with backoff...")
                raise Exception(f"Rate limited: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Submission Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Response text: {response.text[:500]}")
                return None
            
            result = response.json()
            submission_id = result.get('submission_id')
            
            if submission_id:
                print(f"✓ Submission created: {submission_id}")
                return str(submission_id)
            
            return None
            
        except Exception as e:
            print(f"Submission error: {str(e)}")
            return None
    
    def check_submission(self, submission_id: str, max_wait: int = 60) -> Optional[Dict]:
        """
        Poll submission status until complete
        
        Args:
            submission_id: ID of the submission
            max_wait: Maximum time to wait in seconds
        
        Returns:
            Submission result dictionary
        """
        url = f"{Config.LEETCODE_BASE_URL}/submissions/detail/{submission_id}/check/"
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = self.session.get(url)
                result = response.json()
                
                state = result.get('state')
                
                if state == 'SUCCESS':
                    return {
                        'status': result.get('status_msg', 'Unknown'),
                        'runtime': result.get('status_runtime', '0'),
                        'memory': result.get('status_memory', '0'),
                        'runtime_percentile': result.get('runtime_percentile'),
                        'memory_percentile': result.get('memory_percentile'),
                        'total_correct': result.get('total_correct'),
                        'total_testcases': result.get('total_testcases'),
                        'code_output': result.get('code_output', ''),
                        'compile_error': result.get('compile_error', ''),
                        'runtime_error': result.get('runtime_error', ''),
                        'full_runtime_error': result.get('full_runtime_error', ''),
                        'submission_id': submission_id
                    }
                elif state == 'FAILURE':
                    return {
                        'status': 'Error',
                        'error': result.get('status_msg', 'Unknown error'),
                        'submission_id': submission_id
                    }
                
                time.sleep(Config.SUBMISSION_POLL_INTERVAL)
                
            except Exception as e:
                print(f"Error checking submission: {str(e)}")
                time.sleep(Config.SUBMISSION_POLL_INTERVAL)
        
        return {'status': 'Timeout', 'submission_id': submission_id}
    
    def _graphql_request(self, query: str, variables: Dict, operation_name: str = None) -> Optional[Dict]:
        """Make a GraphQL request to LeetCode"""
        headers = {
            'Content-Type': 'application/json',
            'Referer': Config.LEETCODE_BASE_URL,
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Origin': Config.LEETCODE_BASE_URL,
        }
        
        if self.csrf_token:
            headers['X-CSRFToken'] = self.csrf_token
        
        payload = {
            'query': query,
            'variables': variables
        }
        
        if operation_name:
            payload['operationName'] = operation_name
        
        try:
            response = self.session.post(
                Config.LEETCODE_GRAPHQL_URL,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            # Check response before raising
            if response.status_code != 200:
                print(f"GraphQL Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Response text: {response.text[:500]}")
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"GraphQL request error: {str(e)}")
            return None
    
    def get_random_problems(self, count: int = 10, difficulty: Optional[str] = None) -> List[Dict]:
        """
        Get a random selection of problems
        
        Args:
            count: Number of problems to fetch
            difficulty: Optional difficulty filter
        
        Returns:
            List of problem details
        """
        import random
        
        # Fetch a larger pool to randomize from (4x the requested count)
        pool_size = min(count * 4, 200)  # Cap at 200 to avoid excessive API calls
        problems_list = self.get_problem_list(limit=pool_size, difficulty=difficulty)
        
        # Filter out paid-only problems
        free_problems = [p for p in problems_list if not p.get('paidOnly', False)]
        
        # Randomly sample from the pool
        if len(free_problems) > count:
            selected_problems = random.sample(free_problems, count)
        else:
            selected_problems = free_problems
        
        # Fetch detailed information for selected problems
        detailed_problems = []
        for problem in selected_problems:
            details = self.get_problem_details(problem['titleSlug'])
            if details:
                detailed_problems.append(details)
            time.sleep(0.5)  # Rate limiting
        
        return detailed_problems
