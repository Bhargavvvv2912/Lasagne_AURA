import sys
import os

# Bypass legacy path issues
os.environ["THEANO_FLAGS"] = "base_compiledir=/tmp/lasagne,device=cpu"

def test_lasagne_stack():
    try:
        # 1. TEST THE ENGINE (The NumPy 2.0 Tripwire)
        import lasagne
        import theano
        l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28))
        print("✅ Engine Check: Lasagne layers initialized.")

        # 2. TEST THE TEMPLATING (The Jinja2 3.0 Tripwire)
        # Old Sphinx/Lasagne docs logic often calls Jinja2 internals.
        import jinja2
        if not hasattr(jinja2, 'Markup'):
            # In Jinja2 3.1+, Markup was moved to markupsafe.
            # Legacy apps that don't know this will crash here.
            raise AttributeError("Jinja2 'Markup' attribute is missing (3.1+ breakage).")
        
        print("✅ Metadata Check: Jinja2 legacy exports found.")
        return True
        
    except (AttributeError, ImportError, ModuleNotFoundError) as e:
        print(f"❌ Validation Failed: Legacy Stack Crash. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation Failed: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_lasagne_stack()