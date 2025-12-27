from playwright.sync_api import sync_playwright

def verify_frontend():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print("Navigating to http://localhost:5173...")
            page.goto("http://localhost:5173", timeout=60000)
            print("Waiting for 'Trade Viz' header...")
            page.wait_for_selector("text=Trade Viz", timeout=30000)

            # Wait a bit for animations/styles to settle
            page.wait_for_timeout(2000)

            print("Taking screenshot...")
            page.screenshot(path="verification/frontend_viz.png", full_page=True)
            print("Screenshot saved to verification/frontend_viz.png")
        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="verification/error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_frontend()
