# Detailed Exhaustive Orbax Compatibility Plan for ml-switcheroo-compiler

This document provides a truly exhaustive checklist of *every* backend feature, class, and function that `ml-switcheroo-compiler` must implement or support so the `zero-orbax` compiler-frontend can function 100% identically to official `orbax`.

## 1. Checkpoint Handlers

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|
| [ ] | `ArrayCheckpointHandler` | `(checkpoint_name: Optional[str] = None) -> None` | Handler for array checkpoints. | Core compiler struct. |
| [ ] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [ ] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [ ] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [ ] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [ ] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [ ] | `AsyncCheckpointHandler` | `()` | Base class for asynchronous handlers. | Core compiler struct. |
| [ ] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [ ] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [ ] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [ ] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [ ] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [ ] | `BasePyTreeCheckpointHandler` | `(*, save_concurrent_bytes: Optional[int] = None, restore_concurrent_bytes: Op...` | Base handler for PyTree checkpoints. | Core compiler struct. |
| [ ] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [ ] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [ ] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [ ] | `  .get_param_names` | `(self, item: Any) -> Any` | Get parameter names. | Lowered node/operation. |
| [ ] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [ ] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [ ] | `CompositeCheckpointHandler` | `(*item_names: str, composite_options: Any = None, handler_registry: Any = Non...` | Handler for composite checkpoints. | Core compiler struct. |
| [ ] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [ ] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [ ] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [ ] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [ ] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [ ] | `DefaultCheckpointHandlerRegistry` | `(other_registry: Optional[Any] = None) -> None` | Registry for checkpoint handlers. | Core compiler struct. |
| [x] | `  .add` | `(self, item: Optional[str], args: Any, handler: Any) -> None` | Add a handler to the registry. | Lowered node/operation. |
| [ ] | `  .get` | `(self, item: Optional[str], args: Any) -> Any` | Get a handler from the registry. | Lowered node/operation. |
| [ ] | `  .get_all_entries` | `(self) -> Any` | Get all entries. | Lowered node/operation. |
| [ ] | `  .has` | `(self, item: Optional[str], args: Any) -> bool` | Check if an item exists in the registry. | Lowered node/operation. |
| [ ] | `JaxRandomKeyCheckpointHandler` | `(key_name: Optional[str] = None) -> None` | Handler for JAX random keys. | Core compiler struct. |
| [ ] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [ ] | `  .checkpoint_restore_args` | `(self, args: Any) -> Any` | Get restore arguments. | Lowered node/operation. |
| [ ] | `  .checkpoint_save_args` | `(self, args: Any) -> Any` | Get save arguments. | Lowered node/operation. |
| [ ] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [ ] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [ ] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [ ] | `  .post_restore` | `(self, item: Any, metadata: Any) -> Any` | Post-restore hook. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [ ] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [ ] | `JsonCheckpointHandler` | `(filename: Optional[str] = None, *, multiprocessing_options: Any = None) -> None` | Handler for JSON checkpoints. | Core compiler struct. |
| [ ] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [ ] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [ ] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [ ] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [ ] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [ ] | `NumpyRandomKeyCheckpointHandler` | `(key_name: Optional[str] = None) -> None` | Handler for NumPy random keys. | Core compiler struct. |
| [ ] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [ ] | `  .checkpoint_restore_args` | `(self, args: Any) -> Any` | Get restore arguments. | Lowered node/operation. |
| [ ] | `  .checkpoint_save_args` | `(self, args: Any) -> Any` | Get save arguments. | Lowered node/operation. |
| [ ] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [ ] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [ ] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [ ] | `  .post_restore` | `(self, item: Any, metadata: Any) -> Any` | Post-restore hook. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [ ] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [ ] | `ProtoCheckpointHandler` | `(filename: str, *, multiprocessing_options: Any = None) -> None` | Handler for Proto checkpoints. | Core compiler struct. |
| [ ] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [ ] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [ ] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [ ] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [ ] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [ ] | `PyTreeCheckpointHandler` | `(aggregate_filename: Optional[str] = None, *, save_concurrent_gb: Optional[in...` | Handler for PyTree checkpoints. | Core compiler struct. |
| [ ] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [ ] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [ ] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [ ] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [ ] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [ ] | `StandardCheckpointHandler` | `(*, save_concurrent_gb: int = 96, restore_concurrent_gb: int = 96, multiproce...` | Standard checkpoint handler. | Core compiler struct. |
| [ ] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [ ] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [ ] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [ ] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [ ] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
## 2. Checkpointers

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|
| [ ] | `AbstractCheckpointer` | `(*args: Any, **kwargs: Any) -> None` | Abstract base class for saving and restoring items to/from paths. | Core compiler struct. |
| [ ] | `  .restore` | `(self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any` | Restore an item from a given path. | Lowered node/operation. |
| [ ] | `  .save` | `(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any` | Save an item to a given path. | Lowered node/operation. |
| [ ] | `AsyncCheckpointer` | `(_handler=None, *, multiprocessing_options=None, timeout_secs=None, handler=N...` | Checkpointer that performs saves asynchronously. | Core compiler struct. |
| [ ] | `  .restore` | `(self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any) -> '...` | Initiate an asynchronous restore operation. | Lowered node/operation. |
| [ ] | `  .save` | `(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> 'Future'` | Initiate an asynchronous save operation. | Lowered node/operation. |
| [ ] | `  .wait_until_finished` | `(self)` | Block until all background operations are complete. | Lowered node/operation. |
| [ ] | `Checkpointer` | `(handler: zero_orbax.checkpoint.checkpoint_handler.CheckpointHandler, *, mult...` | A standard synchronous checkpointer. | Core compiler struct. |
| [ ] | `  .restore` | `(self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any` | Restore the given item from the specified path synchronously. | Lowered node/operation. |
| [ ] | `  .save` | `(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> None` | Save the given item to the specified path synchronously. | Lowered node/operation. |
| [ ] | `PyTreeCheckpointer` | `(primary_host: Optional[int] = 0, use_ocdbt: bool = True, use_zarr3: bool = F...` | Checkpointer specialized for handling PyTrees. | Core compiler struct. |
| [ ] | `  .restore` | `(self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any` | Restore a PyTree from the specified path. | Lowered node/operation. |
| [ ] | `  .save` | `(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any` | Save a PyTree to the specified path. | Lowered node/operation. |
| [ ] | `StandardCheckpointer` | `(*, async_options=None, multiprocessing_options=None, file_options=None, chec...` | Standard checkpointer for common items. | Core compiler struct. |
| [ ] | `  .restore` | `(self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any` | Restore a common item from the specified path. | Lowered node/operation. |
| [ ] | `  .save` | `(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any` | Save a common item to the specified path. | Lowered node/operation. |
## 3. Checkpoint Management

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|
| [ ] | `AbstractCheckpointManager` | `(*args: Any, **kwargs: Any) -> None` | Abstract base class for a manager that coordinates saving and restoring checkpoints. | Core compiler struct. |
| [ ] | `  .all_steps` | `(self) -> Sequence[int]` | Get all steps with available checkpoints. | Lowered node/operation. |
| [ ] | `  .latest_step` | `(self) -> Optional[int]` | Get the latest saved step. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, step: int, items: Optional[Any] = None, **kwargs: Any) -> Any` | Restore a checkpoint from a specific step. | Lowered node/operation. |
| [ ] | `  .save` | `(self, step: int, items: Any, **kwargs: Any) -> bool` | Save a checkpoint at the given step. | Lowered node/operation. |
| [ ] | `CheckpointManager` | `(directory: Any, checkpointers: Optional[Any] = None, options: 'Optional[Any]...` | Concrete implementation of a manager that coordinates saving and restoring checkpoints. | Core compiler struct. |
| [ ] | `  .all_steps` | `(self) -> Sequence[int]` | Return a sequence of all step numbers that have a checkpoint. | Lowered node/operation. |
| [ ] | `  .latest_step` | `(self) -> Optional[int]` | Return the most recently saved step number. | Lowered node/operation. |
| [ ] | `  .restore` | `(self, step: int, items: Optional[Any] = None, **kwargs: Any) -> Any` | Restore a checkpoint for a specific step. | Lowered node/operation. |
| [ ] | `  .save` | `(self, step: int, items: Any, **kwargs: Any) -> bool` | Save a checkpoint at the given step if conditions are met. | Lowered node/operation. |
| [ ] | `CheckpointManagerOptions` | `(save_interval_steps: int = 1, max_to_keep: Optional[int] = None, keep_time_i...` | Options to configure the behavior of a CheckpointManager. | Core compiler struct. |
| [ ] | `  .replace` | `(self, **kwargs)` | Create a new options instance with updated attributes. | Lowered node/operation. |
## 4. Arguments & Options

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|
| [ ] | `ArrayRestoreArgs` | `(restore_type: Optional[Any] = None, dtype: Optional[Any] = None, mesh: Optio...` | Arguments for restoring an array. | Core compiler struct. |
| [ ] | `AsyncOptions` | `(timeout_secs: int = 300, barrier_sync_fn: Optional[Any] = None, post_finaliz...` | Configuration options for asynchronous checkpointing. | Core compiler struct. |
| [ ] | `SaveArgs` | `(aggregate: bool = False, dtype: Optional[Any] = None, chunk_byte_size: Optio...` | Arguments for saving a checkpoint. | Core compiler struct. |
| [ ] | `args` | `()` | Namespace for various checkpoint argument classes. | Core compiler struct. |
| [ ] | `  .get_registered_args_cls` | `(*a, **kw)` | Get the registered arguments class. | Lowered node/operation. |
| [ ] | `  .get_registered_handler_cls` | `(*a, **kw)` | Get the registered handler class for the given arguments. | Lowered node/operation. |
| [ ] | `  .has_registered_args` | `(*a, **kw)` | Check if there are registered arguments. | Lowered node/operation. |
| [ ] | `  .register_with_handler` | `(*a, **kw)` | Register a handler class. | Lowered node/operation. |
| [ ] | `  .ArrayRestore` | `(*args, **kwargs)` | Arguments for restoring an array. | Inner class support. |
| [ ] | `  .ArraySave` | `(*args, **kwargs)` | Arguments for saving an array. | Inner class support. |
| [ ] | `  .CheckpointArgs` | `()` | Base arguments for a checkpoint handler. | Inner class support. |
| [ ] | `  .Composite` | `(*args, **kwargs)` | Arguments for a composite checkpoint handler. | Inner class support. |
| [ ] | `  .JaxRandomKeyRestore` | `(*args, **kwargs)` | Arguments for restoring a JAX random key. | Inner class support. |
| [ ] | `  .JaxRandomKeySave` | `(*args, **kwargs)` | Arguments for saving a JAX random key. | Inner class support. |
| [ ] | `  .JsonRestore` | `(*args, **kwargs)` | Arguments for restoring from a JSON file. | Inner class support. |
| [ ] | `  .JsonSave` | `(*args, **kwargs)` | Arguments for saving to a JSON file. | Inner class support. |
| [ ] | `  .NumpyRandomKeyRestore` | `(*args, **kwargs)` | Arguments for restoring a NumPy random key. | Inner class support. |
| [ ] | `  .NumpyRandomKeySave` | `(*args, **kwargs)` | Arguments for saving a NumPy random key. | Inner class support. |
| [ ] | `  .ProtoRestore` | `(*args, **kwargs)` | Arguments for restoring from a Protocol Buffer. | Inner class support. |
| [ ] | `  .ProtoSave` | `(*args, **kwargs)` | Arguments for saving to a Protocol Buffer. | Inner class support. |
| [ ] | `  .PyTreeRestore` | `(*args, **kwargs)` | Arguments for restoring a PyTree structure. | Inner class support. |
| [ ] | `  .PyTreeSave` | `(*args, **kwargs)` | Arguments for saving a PyTree structure. | Inner class support. |
| [ ] | `  .StandardRestore` | `(*args, **kwargs)` | Arguments for standard restoration processes. | Inner class support. |
| [ ] | `  .StandardSave` | `(*args, **kwargs)` | Arguments for standard saving processes. | Inner class support. |
## 5. Utilities & Transforms

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|
| [ ] | `Future` | `(result=None, *args: Any, **kwargs: Any) -> None` | A simple representation of an asynchronous result. | Core compiler struct. |
| [ ] | `  .result` | `(self) -> Any` | Retrieve the wrapped result. | Lowered node/operation. |
| [ ] | `NoneType` | `(...)` |  | Core compiler struct. |
| [ ] | `RestoreTransform` | `(value_fn: Optional[Any] = None, multi_value_fn: Optional[Any] = None, multi_...` | Rule defining how to transform data during a restore operation. | Core compiler struct. |
| [ ] | `Transform` | `(original_key: Optional[Any] = None, use_fallback: bool = False, value_fn: Op...` | Defines a general transformation on data values. | Core compiler struct. |
| [ ] | `abstract_checkpoint_manager` | `()` | Namespace for abstract_checkpoint_manager. | Core compiler struct. |
| [ ] | `abstract_checkpointer` | `()` | Namespace for abstract_checkpointer. | Core compiler struct. |
| [ ] | `abstract_logger` | `()` | Namespace for abstract logging interfaces. | Core compiler struct. |
| [ ] | `aggregate_handlers` | `()` | Namespace for aggregate_handlers. | Core compiler struct. |
| [ ] | `async_checkpoint_handler` | `()` | Namespace for asynchronous checkpoint handler functionality. | Core compiler struct. |
| [ ] | `async_checkpointer` | `()` | Namespace for async_checkpointer. | Core compiler struct. |
| [ ] | `async_checkpointer_module` | `()` | Namespace for async_checkpointer. | Core compiler struct. |
| [ ] | `atomicity` | `()` | Namespace for atomic operations and paths. | Core compiler struct. |
| [ ] | `checkpoint` | `()` | Namespace for general checkpoint operations. | Core compiler struct. |
| [ ] | `checkpoint_args` | `()` | Namespace for checkpoint_args. | Core compiler struct. |
| [ ] | `checkpoint_handler` | `()` | Namespace for checkpoint handler. | Core compiler struct. |
| [ ] | `  .CheckpointHandler` | `()` | Base class for handlers that manage checkpoint reading and writing. | Inner class support. |
| [ ] | `checkpoint_manager` | `()` | Namespace for checkpoint_manager. | Core compiler struct. |
| [ ] | `checkpoint_manager_module` | `()` | Namespace for checkpoint_manager. | Core compiler struct. |
| [ ] | `checkpoint_utils` | `()` | Namespace for checkpoint_utils. | Core compiler struct. |
| [ ] | `checkpointer` | `()` | Namespace for checkpointer. | Core compiler struct. |
| [ ] | `checkpointer_module` | `()` | Namespace for checkpointer. | Core compiler struct. |
| [ ] | `epath` | `()` | Namespace for path-like interfaces. | Core compiler struct. |
| [ ] | `future` | `()` | Namespace for future. | Core compiler struct. |
| [ ] | `future_module` | `()` | Namespace for future. | Core compiler struct. |
| [ ] | `handlers` | `()` | Namespace for handlers. | Core compiler struct. |
| [ ] | `metadata` | `()` | Namespace for metadata. | Core compiler struct. |
| [ ] | `msgpack_utils` | `()` | Namespace for msgpack_utils. | Core compiler struct. |
| [ ] | `multihost` | `()` | Namespace for multi-host synchronization. | Core compiler struct. |
| [ ] | `nest_asyncio` | `()` | Namespace for nest_asyncio. | Core compiler struct. |
| [ ] | `options` | `()` | Namespace for orbax checkpointing options. | Core compiler struct. |
| [ ] | `orbax` | `()` | Namespace for orbax classes. | Core compiler struct. |
| [ ] | `  .checkpoint` | `()` | Namespace for orbax checkpointing. | Inner class support. |
| [ ] | `path` | `()` | Namespace for path. | Core compiler struct. |
| [ ] | `pytree_checkpointer` | `()` | Namespace for pytree_checkpointer. | Core compiler struct. |
| [ ] | `pytree_checkpointer_module` | `()` | Namespace for pytree_checkpointer. | Core compiler struct. |
| [ ] | `serialization` | `()` | Namespace for serialization. | Core compiler struct. |
| [ ] | `standard_checkpointer` | `()` | Namespace for standard_checkpointer. | Core compiler struct. |
| [ ] | `standard_checkpointer_module` | `()` | Namespace for standard_checkpointer. | Core compiler struct. |
| [x] | `step` | `()` | Namespace for step-related metadata and formatting. | Core compiler struct. |
| [ ] | `  .standard_name_format` | `(single_host_load_and_broadcast=False, *args, **kwargs) -> Any` | Create a standard step name format. | Lowered node/operation. |
| [ ] | `  .Metadata` | `()` | Metadata associated with a checkpoint step. | Inner class support. |
| [ ] | `  .NameFormat` | `(single_host_load_and_broadcast=False)` | Formatting logic for step names. | Inner class support. |
| [ ] | `step_lib` | `()` | Namespace for step-related metadata and formatting. | Core compiler struct. |
| [ ] | `  .standard_name_format` | `(single_host_load_and_broadcast=False, *args, **kwargs) -> Any` | Create a standard step name format. | Lowered node/operation. |
| [ ] | `  .Metadata` | `()` | Metadata associated with a checkpoint step. | Inner class support. |
| [ ] | `  .NameFormat` | `(single_host_load_and_broadcast=False)` | Formatting logic for step names. | Inner class support. |
| [ ] | `test_utils` | `()` | Namespace for test_utils. | Core compiler struct. |
| [ ] | `transform_utils` | `()` | Namespace for transform_utils. | Core compiler struct. |
| [ ] | `tree` | `()` | Namespace for tree. | Core compiler struct. |
| [ ] | `type_handlers` | `()` | Namespace for type_handlers. | Core compiler struct. |
| [ ] | `utils` | `()` | Namespace for utils. | Core compiler struct. |
| [ ] | `apply_transformations` | `(original_tree: Any, transformations: Any, new_tree: Any, default_to_original...` | Apply a set of transformations to map from an original PyTree to a new PyTree structure. | Graph traversal/utility. |
| [ ] | `merge_trees` | `(*trees, target=None)` | Merge multiple PyTrees into a single PyTree dict. | Graph traversal/utility. |
## 6. Storage & Networking Backend (Internal Primitives)

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|

## 6. Storage & Networking Backend (Internal Primitives)

| Implement | Name | Expected Internal Signature | Description | Backend Notes |
|-----------|------|-----------------------------|-------------|---------------|
| [ ] | `FileSystemIO` | `read(path), write(path, data)` | Epath/TensorStore bindings | Must support standard VFS, GCS, S3 natively. |
| [ ] | `AtomicWrite` | `atomic_write_bytes(path)` | Write to temp file then rename | Cross-platform atomic file replacement. |
| [ ] | `TensorStoreDriver` | `open(spec), read(), write()` | Zarr/Tensorstore chunked IO | Native array chunking and serialization. |
| [ ] | `MsgpackEngine` | `packb(tree), unpackb(bytes)` | Msgpack parser/emitter | Highly optimized C++ or Rust binary parser. |
| [ ] | `BarrierSync` | `wait(id, timeout)` | Multi-host barrier | Sync processes before/after checkpointing. |
| [ ] | `GlobalArraySharding`| `get_local_shard(array, host_id)` | Extract local slice | Support multi-device slice serialization. |
