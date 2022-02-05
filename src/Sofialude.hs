{-# LANGUAGE ImportQualifiedPost #-}
{-# LANGUAGE NoImplicitPrelude #-}

module Sofialude
  ( module Relude.Applicative,
    module Relude.Base,
    module Relude.Bool,
    module Relude.Container,
    module Relude.Debug,
    module Relude.DeepSeq,
    module Relude.Enum,
    module Relude.Exception,
    module Relude.File,
    module Relude.Foldable,
    module Relude.Function,
    module Relude.Functor,
    module Relude.Lifted,
    module Relude.List,
    module Relude.Monad,
    module Relude.Monoid,
    module Relude.Numeric,
    module Relude.Print,
    module Relude.String,
  )
where

import Control.Comonad
import Control.Comonad.Env hiding (ask, asks, local)
import Control.Comonad.Env as Env
import Control.Comonad.Hoist.Class
import Control.Comonad.Identity
import Control.Comonad.Store
import Control.Comonad.Store.Pointer
import Control.Comonad.Store.Zipper hiding (size)
import Control.Comonad.Store.Zipper as StoreZipper
import Control.Comonad.Traced hiding (trace)
import Control.Comonad.Traced as Trace
import Control.Comonad.Trans.Env (Env, EnvT (..), env, lowerEnvT, runEnv, runEnvT)
import Control.Comonad.Trans.Store (Store, StoreT (..), runStore, runStoreT, store)
import Control.Comonad.Trans.Traced hiding (trace)
import Control.Lens
import Control.Monad.Trans.Can
import Control.Monad.Trans.Smash
import Control.Monad.Trans.Wedge
import Data.Align
import Data.Array.IArray (Array, IArray, accum, accumArray, array, assocs, bounds, elems, indices, listArray, (!), (//))
import Data.Array.IArray qualified as IArray
import Data.Array.IO (IOArray, IOUArray, hGetArray, hPutArray)
import Data.Array.MArray.Safe (MArray, freeze, getAssocs, getBounds, getElems, mapArray, mapIndices, newArray, newListArray, readArray, thaw, writeArray)
import Data.Array.ST (STArray, STUArray, runSTArray, runSTUArray)
import Data.Array.Storable (StorableArray)
import Data.Array.Unboxed (UArray)
import Data.Bifunctor.Assoc
import Data.Bifunctor.Biff
import Data.Bifunctor.Clown
import Data.Bifunctor.Flip
import Data.Bifunctor.Join
import Data.Bifunctor.Joker
import Data.Bifunctor.Product
import Data.Bifunctor.Sum
import Data.Bifunctor.Swap
import Data.Bifunctor.Tannen
import Data.Can
  ( Can (..),
    can,
    canCurry,
    canEach,
    canEachA,
    canFst,
    canSnd,
    canUncurry,
    canWithMerge,
    codistributeCan,
    distributeCan,
    foldEnos,
    foldOnes,
    foldTwos,
    isEno,
    isNon,
    isOne,
    isTwo,
    type (⊗),
  )
import Data.Can qualified
import Data.Crosswalk
import Data.Fix hiding (ana, anaM, cata, cataM, hylo, hyloM)
import Data.Functor.Base
import Data.Functor.Combinator hiding (collectI, getI)
import Data.Functor.Foldable hiding (fold)
import Data.Functor.Product qualified as Product
import Data.Functor.Sum qualified as Sum
import Data.Functor.These
import Data.HashMap.Lazy qualified
import Data.HashMap.Strict qualified
import Data.IntMap qualified as IntMap
import Data.Ix
import Data.Map qualified as Map
import Data.Maybe qualified
import Data.Profunctor
import Data.Profunctor.Choice
import Data.Profunctor.Closed
import Data.Profunctor.Mapping
import Data.Profunctor.Ran
import Data.Profunctor.Rep
import Data.Profunctor.Sieve
import Data.Profunctor.Strong
import Data.Profunctor.Traversing
import Data.Profunctor.Yoneda
import Data.Semialign
import Data.Smash
  ( Smash (..),
    distributeSmash,
    foldSmashes,
    fromSmash,
    hulkSmash,
    isNada,
    isSmash,
    pairSmash,
    pairSmashCan,
    quotSmash,
    smash,
    smashCurry,
    smashDiag,
    smashDiag',
    smashFst,
    smashSnd,
    smashUncurry,
    toSmash,
    undistributeSmash,
    unpairSmash,
    unpairSmashCan,
    type (⨳),
  )
import Data.Smash qualified
import Data.Tagged
import Data.These hiding (partitionHereThere, partitionThese)
import Data.These.Combinators hiding (assocThese, bimapThese, bitraverseThese, catHere, catThat, catThere, catThese, catThis, swapThese, unassocThese)
import Data.Time.Calendar
import Data.Time.Clock
import Data.Time.LocalTime
import Data.Validation hiding (fromEither, orElse, toEither)
import Data.Validation qualified as Validation
import Data.Wedge
  ( Wedge (..),
    codistributeWedge,
    distributeWedge,
    foldHeres,
    foldTheres,
    fromWedge,
    isHere,
    isNowhere,
    isThere,
    quotWedge,
    toWedge,
    wedge,
    wedgeLeft,
    wedgeRight,
    type (∨),
  )
import Data.Zip
import Relude.Applicative
import Relude.Base hiding (chr)
import Relude.Bool
import Relude.Container
import Relude.Debug
import Relude.DeepSeq
import Relude.Enum
import Relude.Exception
import Relude.File
import Relude.Foldable
import Relude.Function
import Relude.Functor
import Relude.Lifted
import Relude.List hiding (filter, map, partitionWith, zip, zip3, zipWith, (++))
import Relude.Monad hiding (catMaybes, lefts, mapMaybe, mapMaybeM, partitionEithers, rights)
import Relude.Monoid hiding (Option, WrappedMonoid)
import Relude.Numeric
import Relude.Print
import Relude.String

-- | Alias for 'fmap'
map :: Functor f => (a -> b) -> f a -> f b
map = fmap

-- | Alias for '(<>)'
(++) :: Monoid m => m -> m -> m
(++) = (<>)

mapIArray :: (IArray a e', IArray a e, Ix i) => (e' -> e) -> a i e' -> a i e
mapIArray = IArray.amap

mapIIndices :: (IArray a e, Ix i, Ix j) => (i, i) -> (i -> j) -> a j e -> a i e
mapIIndices = IArray.ixmap

--align :: Semialign f => f a -> f b -> f (These a b)

getIndices :: (MArray a e m, Ix i) => a i e -> m [i]
getIndices arr = do
  b <- getBounds arr
  case b of
    (l, u) -> pure (range (l, u))

-- | Gets the enviroment
askW :: ComonadEnv e w => w a -> e
askW = Env.ask

-- | Gets the environment and applies the specified function
asksW :: ComonadEnv e w => (e -> e') -> w a -> e'
asksW = Env.asks

-- | Modifies the environment using the specified function
localW :: (e -> e') -> EnvT e w a -> EnvT e' w a
localW = Env.local

traceW :: ComonadTraced m w => m -> w a -> a
traceW = Trace.trace

-- TODO: overload?
zipperSize :: Zipper t a -> Int
zipperSize = StoreZipper.size

class Functor f => Filterable f where
  {-# MINIMAL catMaybes | mapMaybe #-}

  filterUnalign :: f (These a b) -> (f (Maybe a), f (Maybe b))
  filterUnalign f = (fmap justHere f, fmap justThere f)

  catMaybes :: f (Maybe a) -> f a
  catMaybes = mapMaybe id

  mapMaybe :: (a -> Maybe b) -> f a -> f b
  mapMaybe f = catMaybes . map f

partitionEithers :: Filterable f => f (Either a b) -> (f a, f b)
partitionEithers = (\(a, b) -> (catMaybes a, catMaybes b)) . filterUnalign . map (either This That)

partitionWith :: Filterable f => (a -> Either b c) -> f a -> (f b, f c)
partitionWith f = partitionEithers . fmap f

filter :: Filterable f => (a -> Bool) -> f a -> f a
filter p = mapMaybe (\x -> if p x then Just x else Nothing)

lefts :: Filterable f => f (Either a b) -> f a
lefts = fst . partitionEithers

rights :: Filterable f => f (Either a b) -> f b
rights = snd . partitionEithers

(<$?>) :: Filterable f => (a -> Maybe b) -> f a -> f b
(<$?>) = mapMaybe

infixl 4 <$?>

(<&?>) :: Filterable f => f a -> (a -> Maybe b) -> f b
as <&?> f = mapMaybe f as

infixl 1 <&?>

partitionThese :: Filterable f => f (These a b) -> (f a, f b, f (a, b))
partitionThese f = (mapMaybe justThis f, mapMaybe justThat f, mapMaybe justThese f)

partitionHereThere :: Filterable f => f (These a b) -> (f a, f b)
partitionHereThere = bimap catMaybes catMaybes . filterUnalign

catThis :: Filterable f => f (These a b) -> f a
catThis = mapMaybe justThis

catThat :: Filterable f => f (These a b) -> f b
catThat = mapMaybe justThat

catThese :: Filterable f => f (These a b) -> f (a, b)
catThese = mapMaybe justThese

catHere :: Filterable f => f (These a b) -> f a
catHere = mapMaybe justHere

catThere :: Filterable f => f (These a b) -> f b
catThere = mapMaybe justThere

instance Filterable [] where
  filterUnalign = foldr (these (\a -> first (Just a :)) (\b -> second (Just b :)) (\a b -> bimap (Just a :) (Just b :))) ([], [])
  catMaybes = Data.Maybe.catMaybes

instance Filterable Maybe where
  filterUnalign Nothing = (Nothing, Nothing)
  filterUnalign (Just (This a)) = (Just (Just a), Just Nothing)
  filterUnalign (Just (That b)) = (Just Nothing, Just (Just b))
  filterUnalign (Just (These a b)) = (Just (Just a), Just (Just b))

  catMaybes = join

instance Filterable IntMap where
  mapMaybe = IntMap.mapMaybe

instance Filterable Seq where
  filterUnalign = bimap fromList fromList . filterUnalign . toList
  mapMaybe f = fromList . mapMaybe f . toList
  catMaybes = fromList . catMaybes . toList

instance Monoid e => Filterable (Either e) where
  mapMaybe f = \case
    Left e -> Left e
    Right a -> maybe (Left mempty) Right (f a)

instance Monoid e => Filterable (Validation e) where
  mapMaybe f = \case
    Failure e -> Failure e
    Success a -> maybe (Failure mempty) Success (f a)

instance Filterable Proxy where
  filterUnalign _ = (Proxy, Proxy)
  catMaybes _ = Proxy
  mapMaybe _ _ = Proxy

instance Filterable (Map k) where
  mapMaybe = Map.mapMaybe

instance Functor f => Filterable (MaybeT f) where
  mapMaybe f = MaybeT . fmap (mapMaybe f) . runMaybeT

instance Filterable ZipList where
  filterUnalign = bimap ZipList ZipList . filterUnalign . getZipList
  catMaybes = ZipList . catMaybes . getZipList
  mapMaybe f = ZipList . mapMaybe f . getZipList

instance Filterable (HashMap k) where
  mapMaybe = Data.HashMap.Lazy.mapMaybe

instance Filterable (Const r) where
  filterUnalign (Const r) = (Const r, Const r)
  catMaybes (Const r) = Const r
  mapMaybe f (Const r) = Const r

instance Filterable f => Filterable (IdentityT f) where
  mapMaybe f (IdentityT a) = IdentityT (mapMaybe f a)

instance (Filterable f, Filterable g) => Filterable (Sum.Sum f g) where
  mapMaybe f (Sum.InL x) = Sum.InL (mapMaybe f x)
  mapMaybe f (Sum.InR x) = Sum.InR (mapMaybe f x)

instance (Filterable f, Filterable g) => Filterable (Product.Product f g) where
  mapMaybe f (Product.Pair x y) = Product.Pair (mapMaybe f x) (mapMaybe f y)

instance (Functor f, Filterable g) => Filterable (Compose f g) where
  mapMaybe f (Compose a) = Compose (fmap (mapMaybe f) a)

class (Traversable t, Filterable t) => Witherable t where
  {-# MINIMAL #-}

  wither :: (Witherable t, Applicative f) => (a -> f (Maybe b)) -> t a -> f (t b)
  wither f = fmap catMaybes . traverse f

witherEither :: (Witherable t, Applicative f) => (a -> f (Either b c)) -> t a -> f (t b, t c)
witherEither f = fmap partitionEithers . traverse f

filterA :: (Witherable t, Applicative f) => (a -> f Bool) -> t a -> f (t a)
filterA f = wither (\x -> f x <&> \b -> if b then Just x else Nothing)

witherMap :: (Witherable t, Applicative m) => (t b -> r) -> (a -> m (Maybe b)) -> t a -> m r
witherMap p f = fmap p . wither f

eitherToValidation :: Either e a -> Validation e a
eitherToValidation = Validation.fromEither

validationToEither :: Validation e a -> Either e a
validationToEither = Validation.toEither

validateOrElse :: Validate v => v e a -> a -> a
validateOrElse = Validation.orElse

instance Witherable []

instance Witherable Maybe

instance Witherable IntMap

instance Witherable Seq

instance Monoid e => Witherable (Either e)

instance Monoid e => Witherable (Validation e)

instance Witherable Proxy

instance Witherable (Map k)

instance Traversable f => Witherable (MaybeT f)

instance Witherable ZipList

instance Witherable (HashMap k)

instance Witherable (Const r)

instance Witherable f => Witherable (IdentityT f)

instance (Witherable f, Witherable g) => Witherable (Sum.Sum f g)

instance (Witherable f, Witherable g) => Witherable (Product.Product f g)

instance (Witherable f, Witherable g) => Witherable (Compose f g)

instance Assoc Can where
  assoc = Data.Can.reassocLR
  unassoc = Data.Can.reassocRL

instance Swap Can where
  swap = Data.Can.swapCan

instance Assoc Smash where
  assoc = Data.Smash.reassocLR
  unassoc = Data.Smash.reassocRL

instance Swap Smash where
  swap = Data.Smash.swapCan

instance Assoc Wedge where
  assoc = Data.Wedge.reassocLR
  unassoc = Data.Wedge.reassocRL

instance Swap Wedge where
  swap = Data.Wedge.swapCan

-- gatherCans :: (Zip f, Functor f, Monoid (f (Can a b))) => Can (f a) (f b) -> f (Can a b)
-- gatherCans Non = mempty
-- gatherCans (One fs) = map One fs
-- gatherCans (Eno fs) = map Eno fs
-- gatherCans (Two fs gs) = zipWith Two fs gs

-- class Unfolding c where
--   unfoldingr :: Alternative f => (b -> c a b) -> b -> f a
--   unfoldingrM :: (Monad m, Alternative f) => (b -> m (c a b)) -> b -> m (f a)
--   iterateUntil :: Alternative f => (b -> c a b) -> b -> f a
--   iterateUntilM :: Monad m => Alternative f => (b -> m (c a b)) -> b -> m (f a)
--   accumUntil :: Alternative f => Monoid b => (b -> c a b) -> f a
--   accumUntilM :: Monad m => Alternative f => Monoid b => (b -> m (c a b)) -> m (f a)

-- instance Unfolding Either where
--   unfoldingr f s = case f s of
--     Left a -> pure a
--     Right b ->

-- classes Applicative Alternative Eq Ord Generic Show Read
-- One Hashable IsList NFData Enum Bounded Foldable Traversable Bifoldable Bitraverasable
-- Functor Bifunctor Semigroup Monoid

-- containers HashMap HashSet IntMap IntSet Map Set Seq NonEmpty ByteString Text []

-- critiques
-- Enum is non-total
-- File has locale-sensitive functions
-- FilePath is just wrong
-- Num hierarchy
-- cycle1

-- todo
-- module Relude.Nub via witherable
-- sorts, groups, nubs using discrimination
-- dropWhile etc using witherable
