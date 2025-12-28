from playwright.sync_api import sync_playwright

def verify_frontend():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # Navigate to the app
            page.goto("http://localhost:5173")

            # Wait for content
            page.wait_for_selector(".font-bold")

            # Check if Lab button is GONE from header (left sidebar header)
            # The header has "TradeViz" text.
            # We want to make sure there is no "Lab" or "Microscope" button in the header.
            # The code I removed:
            # <button onClick={() => setCurrentPage('lab')} ...> ... </button>
            # It had a title="Strategy Lab" or text "Lab".

            lab_btn = page.query_selector('button[title="Strategy Lab"]')
            if lab_btn:
                print("FAILURE: Lab button found in header!")
            else:
                print("SUCCESS: Lab button not found in header.")

            # Take a screenshot of the main page
            page.screenshot(path="verification/main_page.png")

            # Verify Agent Terminal exists
            terminal = page.wait_for_selector("text=Agent Terminal")
            if terminal:
                 print("SUCCESS: Agent Terminal found.")

            # Verify new Expander button on Chat
            # It's absolute positioned top right of chat container
            expand_btn = page.query_selector('button[title="Expand"]')
            if expand_btn:
                print("SUCCESS: Expand button found.")
                expand_btn.click()
                page.wait_for_timeout(500) # Wait for transition
                page.screenshot(path="verification/chat_expanded.png")
            else:
                print("WARNING: Expand button not found (might require hover or specific state).")
                # Attempt to find by SVG path if title not set correctly?
                # I set title="Expand" in the code: title={isChatExpanded ? "Collapse" : "Expand"}

        except Exception as e:
            print(f"Error: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    verify_frontend()
