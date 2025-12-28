from playwright.sync_api import sync_playwright

def verify_frontend():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the app
        print("Navigating to http://localhost:5173")
        try:
            page.goto("http://localhost:5173", timeout=30000)

            # Wait for text "TradeViz" which is in the header
            print("Waiting for dashboard to load...")
            page.wait_for_selector("text=TradeViz", state='visible', timeout=10000)

            # Take screenshot of the main page
            print("Taking dashboard screenshot...")
            page.screenshot(path="verification/dashboard_view.png", full_page=True)
            print("Dashboard screenshot taken.")

            # Interact with sidebar - verify Navigator
            print("Switching to Trades view...")
            trades_btn = page.locator("button:has-text('Trades')").first
            if trades_btn.is_visible():
                trades_btn.click()
                page.wait_for_timeout(1000)
                page.screenshot(path="verification/trades_view.png")
                print("Trades view screenshot taken.")
            else:
                print("Trades button not found!")

        except Exception as e:
            print(f"Error occurred: {e}")
            try:
                page.screenshot(path="verification/final_error.png")
            except:
                pass
        finally:
            browser.close()

if __name__ == "__main__":
    verify_frontend()
