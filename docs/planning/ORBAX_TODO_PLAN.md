# Detailed Exhaustive Orbax Compatibility Plan for ml-switcheroo-compiler

This document provides a truly exhaustive checklist of *every* backend feature, class, and function that `ml-switcheroo-compiler` must implement or support so the `zero-orbax` compiler-frontend can function 100% identically to official `orbax`.

## 1. Checkpoint Handlers

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|
| [x] | `ArrayCheckpointHandler` | `(checkpoint_name: Optional[str] = None) -> None` | Handler for array checkpoints. | Core compiler struct. |
| [x] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [x] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [x] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [x] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [x] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [x] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [x] | `AsyncCheckpointHandler` | `()` | Base class for asynchronous handlers. | Core compiler struct. |
| [x] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [x] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [x] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [x] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [x] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [x] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [x] | `BasePyTreeCheckpointHandler` | `(*, save_concurrent_bytes: Optional[int] = None, restore_concurrent_bytes: Op...` | Base handler for PyTree checkpoints. | Core compiler struct. |
| [x] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [x] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [x] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [x] | `  .get_param_names` | `(self, item: Any) -> Any` | Get parameter names. | Lowered node/operation. |
| [x] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [x] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [x] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [x] | `CompositeCheckpointHandler` | `(*item_names: str, composite_options: Any = None, handler_registry: Any = Non...` | Handler for composite checkpoints. | Core compiler struct. |
| [x] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [x] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [x] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [x] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [x] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [x] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [x] | `DefaultCheckpointHandlerRegistry` | `(other_registry: Optional[Any] = None) -> None` | Registry for checkpoint handlers. | Core compiler struct. |
| [x] | `  .add` | `(self, item: Optional[str], args: Any, handler: Any) -> None` | Add a handler to the registry. | Lowered node/operation. |
| [x] | `  .get` | `(self, item: Optional[str], args: Any) -> Any` | Get a handler from the registry. | Lowered node/operation. |
| [x] | `  .get_all_entries` | `(self) -> Any` | Get all entries. | Lowered node/operation. |
| [x] | `  .has` | `(self, item: Optional[str], args: Any) -> bool` | Check if an item exists in the registry. | Lowered node/operation. |
| [x] | `JaxRandomKeyCheckpointHandler` | `(key_name: Optional[str] = None) -> None` | Handler for JAX random keys. | Core compiler struct. |
| [x] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [x] | `  .checkpoint_restore_args` | `(self, args: Any) -> Any` | Get restore arguments. | Lowered node/operation. |
| [x] | `  .checkpoint_save_args` | `(self, args: Any) -> Any` | Get save arguments. | Lowered node/operation. |
| [x] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [x] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [x] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [x] | `  .post_restore` | `(self, item: Any, metadata: Any) -> Any` | Post-restore hook. | Lowered node/operation. |
| [x] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [x] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [x] | `JsonCheckpointHandler` | `(filename: Optional[str] = None, *, multiprocessing_options: Any = None) -> None` | Handler for JSON checkpoints. | Core compiler struct. |
| [x] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [x] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [x] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [x] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [x] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [x] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [x] | `NumpyRandomKeyCheckpointHandler` | `(key_name: Optional[str] = None) -> None` | Handler for NumPy random keys. | Core compiler struct. |
| [x] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [x] | `  .checkpoint_restore_args` | `(self, args: Any) -> Any` | Get restore arguments. | Lowered node/operation. |
| [x] | `  .checkpoint_save_args` | `(self, args: Any) -> Any` | Get save arguments. | Lowered node/operation. |
| [x] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [x] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [x] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [x] | `  .post_restore` | `(self, item: Any, metadata: Any) -> Any` | Post-restore hook. | Lowered node/operation. |
| [x] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [x] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [x] | `ProtoCheckpointHandler` | `(filename: str, *, multiprocessing_options: Any = None) -> None` | Handler for Proto checkpoints. | Core compiler struct. |
| [x] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [x] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [x] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [x] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [x] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [x] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [x] | `PyTreeCheckpointHandler` | `(aggregate_filename: Optional[str] = None, *, save_concurrent_gb: Optional[in...` | Handler for PyTree checkpoints. | Core compiler struct. |
| [x] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [x] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [x] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [x] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [x] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [x] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
| [x] | `StandardCheckpointHandler` | `(*, save_concurrent_gb: int = 96, restore_concurrent_gb: int = 96, multiproce...` | Standard checkpoint handler. | Core compiler struct. |
| [x] | `  .async_save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Optional[List[Any]]` | Asynchronously save an item. | Lowered node/operation. |
| [x] | `  .close` | `(self) -> None` | Close the handler. | Lowered node/operation. |
| [x] | `  .finalize` | `(self, directory: Any) -> None` | Finalize the checkpoint. | Lowered node/operation. |
| [x] | `  .metadata` | `(self, directory: Any) -> Optional[Any]` | Get metadata. | Lowered node/operation. |
| [x] | `  .restore` | `(self, directory: Any, *args: Any, **kwargs: Any) -> Any` | Restore an item. | Lowered node/operation. |
| [x] | `  .save` | `(self, directory: Any, *args: Any, **kwargs: Any) -> None` | Save an item synchronously. | Lowered node/operation. |
## 2. Checkpointers

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|
| [x] | `AbstractCheckpointer` | `(*args: Any, **kwargs: Any) -> None` | Abstract base class for saving and restoring items to/from paths. | Core compiler struct. |
| [x] | `  .restore` | `(self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any` | Restore an item from a given path. | Lowered node/operation. |
| [x] | `  .save` | `(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any` | Save an item to a given path. | Lowered node/operation. |
| [x] | `AsyncCheckpointer` | `(_handler=None, *, multiprocessing_options=None, timeout_secs=None, handler=N...` | Checkpointer that performs saves asynchronously. | Core compiler struct. |
| [x] | `  .restore` | `(self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any) -> '...` | Initiate an asynchronous restore operation. | Lowered node/operation. |
| [x] | `  .save` | `(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> 'Future'` | Initiate an asynchronous save operation. | Lowered node/operation. |
| [x] | `  .wait_until_finished` | `(self)` | Block until all background operations are complete. | Lowered node/operation. |
| [x] | `Checkpointer` | `(handler: zero_orbax.checkpoint.checkpoint_handler.CheckpointHandler, *, mult...` | A standard synchronous checkpointer. | Core compiler struct. |
| [x] | `  .restore` | `(self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any` | Restore the given item from the specified path synchronously. | Lowered node/operation. |
| [x] | `  .save` | `(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> None` | Save the given item to the specified path synchronously. | Lowered node/operation. |
| [x] | `PyTreeCheckpointer` | `(primary_host: Optional[int] = 0, use_ocdbt: bool = True, use_zarr3: bool = F...` | Checkpointer specialized for handling PyTrees. | Core compiler struct. |
| [x] | `  .restore` | `(self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any` | Restore a PyTree from the specified path. | Lowered node/operation. |
| [x] | `  .save` | `(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any` | Save a PyTree to the specified path. | Lowered node/operation. |
| [x] | `StandardCheckpointer` | `(*, async_options=None, multiprocessing_options=None, file_options=None, chec...` | Standard checkpointer for common items. | Core compiler struct. |
| [x] | `  .restore` | `(self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any) -> Any` | Restore a common item from the specified path. | Lowered node/operation. |
| [x] | `  .save` | `(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any` | Save a common item to the specified path. | Lowered node/operation. |
## 3. Checkpoint Management

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|
| [x] | `AbstractCheckpointManager` | `(*args: Any, **kwargs: Any) -> None` | Abstract base class for a manager that coordinates saving and restoring checkpoints. | Core compiler struct. |
| [x] | `  .all_steps` | `(self) -> Sequence[int]` | Get all steps with available checkpoints. | Lowered node/operation. |
| [x] | `  .latest_step` | `(self) -> Optional[int]` | Get the latest saved step. | Lowered node/operation. |
| [x] | `  .restore` | `(self, step: int, items: Optional[Any] = None, **kwargs: Any) -> Any` | Restore a checkpoint from a specific step. | Lowered node/operation. |
| [x] | `  .save` | `(self, step: int, items: Any, **kwargs: Any) -> bool` | Save a checkpoint at the given step. | Lowered node/operation. |
| [x] | `CheckpointManager` | `(directory: Any, checkpointers: Optional[Any] = None, options: 'Optional[Any]...` | Concrete implementation of a manager that coordinates saving and restoring checkpoints. | Core compiler struct. |
| [x] | `  .all_steps` | `(self) -> Sequence[int]` | Return a sequence of all step numbers that have a checkpoint. | Lowered node/operation. |
| [x] | `  .latest_step` | `(self) -> Optional[int]` | Return the most recently saved step number. | Lowered node/operation. |
| [x] | `  .restore` | `(self, step: int, items: Optional[Any] = None, **kwargs: Any) -> Any` | Restore a checkpoint for a specific step. | Lowered node/operation. |
| [x] | `  .save` | `(self, step: int, items: Any, **kwargs: Any) -> bool` | Save a checkpoint at the given step if conditions are met. | Lowered node/operation. |
| [x] | `CheckpointManagerOptions` | `(save_interval_steps: int = 1, max_to_keep: Optional[int] = None, keep_time_i...` | Options to configure the behavior of a CheckpointManager. | Core compiler struct. |
| [x] | `  .replace` | `(self, **kwargs)` | Create a new options instance with updated attributes. | Lowered node/operation. |
## 4. Arguments & Options

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|
| [x] | `ArrayRestoreArgs` | `(restore_type: Optional[Any] = None, dtype: Optional[Any] = None, mesh: Optio...` | Arguments for restoring an array. | Core compiler struct. |
| [x] | `AsyncOptions` | `(timeout_secs: int = 300, barrier_sync_fn: Optional[Any] = None, post_finaliz...` | Configuration options for asynchronous checkpointing. | Core compiler struct. |
| [x] | `SaveArgs` | `(aggregate: bool = False, dtype: Optional[Any] = None, chunk_byte_size: Optio...` | Arguments for saving a checkpoint. | Core compiler struct. |
| [x] | `args` | `()` | Namespace for various checkpoint argument classes. | Core compiler struct. |
| [x] | `  .get_registered_args_cls` | `(*a, **kw)` | Get the registered arguments class. | Lowered node/operation. |
| [x] | `  .get_registered_handler_cls` | `(*a, **kw)` | Get the registered handler class for the given arguments. | Lowered node/operation. |
| [x] | `  .has_registered_args` | `(*a, **kw)` | Check if there are registered arguments. | Lowered node/operation. |
| [x] | `  .register_with_handler` | `(*a, **kw)` | Register a handler class. | Lowered node/operation. |
| [x] | `  .ArrayRestore` | `(*args, **kwargs)` | Arguments for restoring an array. | Inner class support. |
| [x] | `  .ArraySave` | `(*args, **kwargs)` | Arguments for saving an array. | Inner class support. |
| [x] | `  .CheckpointArgs` | `()` | Base arguments for a checkpoint handler. | Inner class support. |
| [x] | `  .Composite` | `(*args, **kwargs)` | Arguments for a composite checkpoint handler. | Inner class support. |
| [x] | `  .JaxRandomKeyRestore` | `(*args, **kwargs)` | Arguments for restoring a JAX random key. | Inner class support. |
| [x] | `  .JaxRandomKeySave` | `(*args, **kwargs)` | Arguments for saving a JAX random key. | Inner class support. |
| [x] | `  .JsonRestore` | `(*args, **kwargs)` | Arguments for restoring from a JSON file. | Inner class support. |
| [x] | `  .JsonSave` | `(*args, **kwargs)` | Arguments for saving to a JSON file. | Inner class support. |
| [x] | `  .NumpyRandomKeyRestore` | `(*args, **kwargs)` | Arguments for restoring a NumPy random key. | Inner class support. |
| [x] | `  .NumpyRandomKeySave` | `(*args, **kwargs)` | Arguments for saving a NumPy random key. | Inner class support. |
| [x] | `  .ProtoRestore` | `(*args, **kwargs)` | Arguments for restoring from a Protocol Buffer. | Inner class support. |
| [x] | `  .ProtoSave` | `(*args, **kwargs)` | Arguments for saving to a Protocol Buffer. | Inner class support. |
| [x] | `  .PyTreeRestore` | `(*args, **kwargs)` | Arguments for restoring a PyTree structure. | Inner class support. |
| [x] | `  .PyTreeSave` | `(*args, **kwargs)` | Arguments for saving a PyTree structure. | Inner class support. |
| [x] | `  .StandardRestore` | `(*args, **kwargs)` | Arguments for standard restoration processes. | Inner class support. |
| [x] | `  .StandardSave` | `(*args, **kwargs)` | Arguments for standard saving processes. | Inner class support. |
## 5. Utilities & Transforms

| Implement | Name | Signature | Docstring | Backend Notes & Compiler Requirements |
|-----------|------|-----------|-----------|-------------------------------------|
| [x] | `Future` | `(result=None, *args: Any, **kwargs: Any) -> None` | A simple representation of an asynchronous result. | Core compiler struct. |
| [x] | `  .result` | `(self) -> Any` | Retrieve the wrapped result. | Lowered node/operation. |
| [x] | `NoneType` | `(...)` |  | Core compiler struct. |
| [x] | `RestoreTransform` | `(value_fn: Optional[Any] = None, multi_value_fn: Optional[Any] = None, multi_...` | Rule defining how to transform data during a restore operation. | Core compiler struct. |
| [x] | `Transform` | `(original_key: Optional[Any] = None, use_fallback: bool = False, value_fn: Op...` | Defines a general transformation on data values. | Core compiler struct. |
| [x] | `abstract_checkpoint_manager` | `()` | Namespace for abstract_checkpoint_manager. | Core compiler struct. |
| [x] | `abstract_checkpointer` | `()` | Namespace for abstract_checkpointer. | Core compiler struct. |
| [x] | `abstract_logger` | `()` | Namespace for abstract logging interfaces. | Core compiler struct. |
| [x] | `aggregate_handlers` | `()` | Namespace for aggregate_handlers. | Core compiler struct. |
| [x] | `async_checkpoint_handler` | `()` | Namespace for asynchronous checkpoint handler functionality. | Core compiler struct. |
| [x] | `async_checkpointer` | `()` | Namespace for async_checkpointer. | Core compiler struct. |
| [x] | `async_checkpointer_module` | `()` | Namespace for async_checkpointer. | Core compiler struct. |
| [x] | `atomicity` | `()` | Namespace for atomic operations and paths. | Core compiler struct. |
| [x] | `checkpoint` | `()` | Namespace for general checkpoint operations. | Core compiler struct. |
| [x] | `checkpoint_args` | `()` | Namespace for checkpoint_args. | Core compiler struct. |
| [x] | `checkpoint_handler` | `()` | Namespace for checkpoint handler. | Core compiler struct. |
| [x] | `  .CheckpointHandler` | `()` | Base class for handlers that manage checkpoint reading and writing. | Inner class support. |
| [x] | `checkpoint_manager` | `()` | Namespace for checkpoint_manager. | Core compiler struct. |
| [x] | `checkpoint_manager_module` | `()` | Namespace for checkpoint_manager. | Core compiler struct. |
| [x] | `checkpoint_utils` | `()` | Namespace for checkpoint_utils. | Core compiler struct. |
| [x] | `checkpointer` | `()` | Namespace for checkpointer. | Core compiler struct. |
| [x] | `checkpointer_module` | `()` | Namespace for checkpointer. | Core compiler struct. |
| [x] | `epath` | `()` | Namespace for path-like interfaces. | Core compiler struct. |
| [x] | `future` | `()` | Namespace for future. | Core compiler struct. |
| [x] | `future_module` | `()` | Namespace for future. | Core compiler struct. |
| [x] | `handlers` | `()` | Namespace for handlers. | Core compiler struct. |
| [x] | `metadata` | `()` | Namespace for metadata. | Core compiler struct. |
| [x] | `msgpack_utils` | `()` | Namespace for msgpack_utils. | Core compiler struct. |
| [x] | `multihost` | `()` | Namespace for multi-host synchronization. | Core compiler struct. |
| [x] | `nest_asyncio` | `()` | Namespace for nest_asyncio. | Core compiler struct. |
| [x] | `options` | `()` | Namespace for orbax checkpointing options. | Core compiler struct. |
| [x] | `orbax` | `()` | Namespace for orbax classes. | Core compiler struct. |
| [x] | `  .checkpoint` | `()` | Namespace for orbax checkpointing. | Inner class support. |
| [x] | `path` | `()` | Namespace for path. | Core compiler struct. |
| [x] | `pytree_checkpointer` | `()` | Namespace for pytree_checkpointer. | Core compiler struct. |
| [x] | `pytree_checkpointer_module` | `()` | Namespace for pytree_checkpointer. | Core compiler struct. |
| [x] | `serialization` | `()` | Namespace for serialization. | Core compiler struct. |
| [x] | `standard_checkpointer` | `()` | Namespace for standard_checkpointer. | Core compiler struct. |
| [x] | `standard_checkpointer_module` | `()` | Namespace for standard_checkpointer. | Core compiler struct. |
| [x] | `step` | `()` | Namespace for step-related metadata and formatting. | Core compiler struct. |
| [x] | `  .standard_name_format` | `(single_host_load_and_broadcast=False, *args, **kwargs) -> Any` | Create a standard step name format. | Lowered node/operation. |
| [x] | `  .Metadata` | `()` | Metadata associated with a checkpoint step. | Inner class support. |
| [x] | `  .NameFormat` | `(single_host_load_and_broadcast=False)` | Formatting logic for step names. | Inner class support. |
| [x] | `step_lib` | `()` | Namespace for step-related metadata and formatting. | Core compiler struct. |
| [x] | `  .standard_name_format` | `(single_host_load_and_broadcast=False, *args, **kwargs) -> Any` | Create a standard step name format. | Lowered node/operation. |
| [x] | `  .Metadata` | `()` | Metadata associated with a checkpoint step. | Inner class support. |
| [x] | `  .NameFormat` | `(single_host_load_and_broadcast=False)` | Formatting logic for step names. | Inner class support. |
| [x] | `test_utils` | `()` | Namespace for test_utils. | Core compiler struct. |
| [x] | `transform_utils` | `()` | Namespace for transform_utils. | Core compiler struct. |
| [x] | `tree` | `()` | Namespace for tree. | Core compiler struct. |
| [x] | `type_handlers` | `()` | Namespace for type_handlers. | Core compiler struct. |
| [x] | `utils` | `()` | Namespace for utils. | Core compiler struct. |
| [x] | `apply_transformations` | `(original_tree: Any, transformations: Any, new_tree: Any, default_to_original...` | Apply a set of transformations to map from an original PyTree to a new PyTree structure. | Graph traversal/utility. |
| [x] | `merge_trees` | `(*trees, target=None)` | Merge multiple PyTrees into a single PyTree dict. | Graph traversal/utility. |
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
