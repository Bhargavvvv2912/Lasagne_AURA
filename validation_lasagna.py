import sys
import os

os.environ["THEANO_FLAGS"] = "base_compiledir=/tmp/lasagne,device=cpu"

def test_lasagne_stack():
    try:
        # 1. TEST THE ENGINE (NumPy 2.0 Tripwire)
        import lasagne
        import theano
        l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28))
        print("✅ Engine Check: Lasagne layers initialized.")

        # 2. TEST THE METADATA (Jinja2/MarkupSafe Tripwire)
        import jinja2
        # This triggers the 'soft_unicode' or 'Markup' check internally
        t = jinja2.Template('Hello {{ name }}!')
        render = t.render(name='ASE')
        
        print("✅ Metadata Check: Jinja2/MarkupSafe stack is functional.")
        return True
        
    except (AttributeError, ImportError, ModuleNotFoundError) as e:
        print(f"❌ Validation Failed: Legacy Stack Crash. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation Failed: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_lasagne_stack()